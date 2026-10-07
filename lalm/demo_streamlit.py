"""Streamlit recording/commit/history UI; model decoding remains upstream."""
import hashlib
import json
import os
from pathlib import Path

import streamlit as st

from demo_engine import DemoEngine, read_audio


st.set_page_config(page_title="Русская речь · ASR и Qwen", page_icon="🎙️", layout="wide")
st.title("🎙️ Русская речь: ASR → коммит → Qwen")
st.caption("Запишите реплику: сначала обычная расшифровка, после коммита — аудиовход Qwen с историей.")
checkpoint = os.environ["IFAO_DEMO_MODEL"]

if os.environ.get("IFAO_DEMO_EXAMPLES"):
    replay_path = Path(os.environ["IFAO_DEMO_EXAMPLES"]).with_name("audio-history-rollout.json")
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text())
        with st.expander("Весь звонок — человек на записи, новые ответы Qwen", expanded=True):
            st.caption(replay["description"])
            st.caption(f'Чекпойнт: {replay["checkpoint"]}. Готово шагов: {len(replay["steps"])}.')
            with st.expander("Системный промпт этого перепрогона"):
                st.write(replay["system"])
            step = st.selectbox("Шаг перепрогона", replay["steps"],
                                format_func=lambda item: f'{item["step"]:02d} · {item["end"]:.1f} с')
            for audio_path in step["audio_paths"]:
                st.audio(Path(audio_path).read_bytes(), format="audio/wav")
            st.write("**GigaAM — только сравнение**", step["gigaam"]["text"])
            st.write("**Qwen по аудио с аудиоисторией**", step["audio_answer"]["text"])
            st.write("**Каскад GigaAM → Qwen с собственной текстовой историей**", step["text_answer"]["text"])
            if step["audio_answer"]["limit_reached"] or step["text_answer"]["limit_reached"]:
                st.warning("Ответ на этом шаге достиг ограничения 256 токенов и мог оборваться.")
            st.caption(f'Аудиореплик в контексте: {step["audio_answer"]["audio_turns"]}; '
                       f'токенов контекста: {step["audio_answer"]["input_tokens"]}.')
            with st.expander("Все новые ответы по порядку"):
                for item in replay["steps"]:
                    st.write(f'**Шаг {item["step"]}** · GigaAM для сравнения: {item["gigaam"]["text"]}')
                    st.write(item["audio_answer"]["text"])


@st.cache_resource
def engine():
    return DemoEngine(checkpoint, os.environ["IFAO_DEMO_RNNT"])


st.session_state.setdefault("history", [])
st.session_state.setdefault("turns", [])
st.session_state.setdefault("recording_id", 0)
st.session_state.setdefault("system_prompt", Path(__file__).with_name("configs").joinpath("system_ru.txt").read_text().strip())
st.session_state.setdefault("use_history", True)


def load_example_context(item):
    st.session_state.history = [dict(turn) for turn in item["history"]]
    st.session_state.turns = []
    st.session_state.system_prompt = item["system"]
    st.session_state.use_history = True
    st.session_state.pop("pending", None)


def load_selected_example(key):
    item = st.session_state[key]
    if item is not None and item.get("auto_context", False):
        load_example_context(item)


with st.sidebar:
    st.write("**Чекпойнт**", Path(checkpoint).name)
    st.caption(os.environ.get("IFAO_DEMO_LABEL", checkpoint))
    st.caption("BF16 · одна GPU · экспериментальная модель")
    system = st.text_area("Системный промпт", key="system_prompt")
    use_history = st.checkbox("Учитывать историю", key="use_history")
    compare_text = st.checkbox("Сравнить с ответом по обычному ASR-тексту", True)
    channel = st.selectbox("Канал загруженного аудио", [0, 1], help="Для стерео выберите канал; голоса не смешиваются автоматически.")
    if st.button("Новый разговор"):
        st.session_state.history = []
        st.session_state.turns = []
        st.session_state.recording_id += 1
        st.session_state.pop("pending", None)
        st.rerun()
    st.caption("Реплики 0,5–30 секунд. Распознавание начинается после остановки записи; это пока не потоковый ASR.")
    st.caption("Подавление чужого голоса ещё проверяется. Ошибки и выдуманные слова возможны.")

history = list(st.session_state.history) if use_history else []
with st.expander(f"История, передаваемая Qwen — {len(history)} сообщений", expanded=True):
    for turn in history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
    if not history:
        st.caption("Qwen получит текущую реплику без истории разговора.")

recording = st.audio_input("Записать реплику", sample_rate=16000, key=f"mic-{st.session_state.recording_id}")
uploaded = st.file_uploader("Или загрузить WAV / FLAC", type=["wav", "flac"], key=f"file-{st.session_state.recording_id}")
source = uploaded if uploaded is not None else recording
example = None
if os.environ.get("IFAO_DEMO_EXAMPLES"):
    examples = json.loads(Path(os.environ["IFAO_DEMO_EXAMPLES"]).read_text())
    example = st.selectbox(
        "Готовая проверка — вместо микрофона или файла", [None, *examples],
        format_func=lambda item: "Использовать микрофон / файл" if item is None else item["name"],
        key=f"example-{st.session_state.recording_id}",
        on_change=load_selected_example, args=(f"example-{st.session_state.recording_id}",),
    )
    if example is not None:
        st.caption(example["description"])
        if "history" in example:
            st.button("Начать с истории примера", key=f"context-{example['name']}",
                      on_click=load_example_context, args=(example,))
        with st.expander("Эталон для сравнения — не передаётся модели"):
            st.write(example["reference"])
if source is not None or example is not None:
    payload = Path(example["audio_path"]).read_bytes() if example is not None else source.getvalue()
    selected_channel = channel if uploaded is not None and example is None else 0
    identity = hashlib.sha256(payload + bytes([selected_channel])).hexdigest()
    if st.session_state.get("pending", {}).get("id") != identity:
        try:
            audio = read_audio(payload, selected_channel)
            with st.spinner("Обычный GigaAM распознаёт запись…"):
                baseline = engine().baseline(audio)
            st.session_state.pending = {"id": identity, "audio": audio, "baseline": baseline}
        except Exception as error:
            st.error(f"Не удалось распознать запись: {error}")
            st.stop()
    pending = st.session_state.pending
    st.audio(pending["audio"], sample_rate=16000)
    st.write("**До коммита · обычный GigaAM**")
    st.write(pending["baseline"]["text"] or "∅ Пустая расшифровка")
    st.caption(f'{pending["baseline"]["seconds"]:.2f} с')
    if st.button("Коммит — передать аудио в Qwen", type="primary"):
        pending.pop("result", None)
        pending.pop("committed", None)
        try:
            history = list(st.session_state.history) if use_history else []
            with st.spinner("Qwen: расшифровка и ответ с учётом истории…"):
                result = {"baseline": pending["baseline"], "history": history,
                          "system": system, "checkpoint": checkpoint}
                result["asr"] = engine().generate(pending["audio"], history, system, asr=True)
                result["answer"] = engine().generate(pending["audio"], history, system)
                if compare_text:
                    result["text_answer"] = engine().generate(None, history, system, text=pending["baseline"]["text"])
            pending["result"] = result
            pending["committed"] = False
        except Exception as error:
            st.error(f"Коммит не выполнен: {error}")
    if "result" in pending:
        result = pending["result"]
        left, right = st.columns(2)
        with left:
            st.write("**После проектора и Qwen · ASR-голова**")
            st.write(result["asr"]["text"] or "∅")
            st.caption(f'{result["asr"]["seconds"]:.2f} с')
            st.write("**Ответ Qwen по аудио**")
            st.write(result["answer"]["text"] or "∅")
        with right:
            if "text_answer" in result:
                st.write("**Ответ той же Qwen по обычному ASR-тексту**")
                st.write(result["text_answer"]["text"] or "∅")
        if any(value.get("limit_reached") for value in result.values() if isinstance(value, dict)):
            st.warning("Один из ответов достиг ограничения длины и мог оборваться.")
        result_id = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        user_text = st.text_area("Текст реплики для следующей истории — можно исправить", result["asr"]["text"], key=f"history-{result_id}")
        if st.button("Добавить реплику и ответ в историю", disabled=pending.get("committed", False)):
            st.session_state.history.extend([{"role": "user", "content": user_text},
                                             {"role": "assistant", "content": result["answer"]["text"]}])
            st.session_state.turns.append(result)
            pending["committed"] = True
            st.rerun()
        st.download_button("Скачать сравнение JSON", json.dumps(result, ensure_ascii=False, indent=2), "comparison.json", "application/json")
