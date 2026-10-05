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


@st.cache_resource
def engine():
    return DemoEngine(checkpoint, os.environ["IFAO_DEMO_RNNT"])


st.session_state.setdefault("history", [])
st.session_state.setdefault("turns", [])
st.session_state.setdefault("recording_id", 0)
with st.sidebar:
    st.write("**Чекпойнт**", Path(checkpoint).name)
    st.caption(os.environ.get("IFAO_DEMO_LABEL", checkpoint))
    st.caption("BF16 · одна GPU · экспериментальная модель")
    system = st.text_area("Системный промпт", Path(__file__).with_name("configs").joinpath("system_ru.txt").read_text())
    use_history = st.checkbox("Учитывать историю", True)
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

recording = st.audio_input("Записать реплику", sample_rate=16000, key=f"mic-{st.session_state.recording_id}")
uploaded = st.file_uploader("Или загрузить WAV / FLAC", type=["wav", "flac"], key=f"file-{st.session_state.recording_id}")
source = uploaded if uploaded is not None else recording
if source is not None:
    payload = source.getvalue()
    selected_channel = channel if uploaded is not None else 0
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
        user_text = st.text_area("Текст реплики для следующей истории — можно исправить", result["asr"]["text"], key=f"history-{identity}")
        if st.button("Добавить реплику и ответ в историю", disabled=pending.get("committed", False)):
            st.session_state.history.extend([{"role": "user", "content": user_text},
                                             {"role": "assistant", "content": result["answer"]["text"]}])
            st.session_state.turns.append(result)
            pending["committed"] = True
            st.rerun()
        st.download_button("Скачать сравнение JSON", json.dumps(result, ensure_ascii=False, indent=2), "comparison.json", "application/json")

with st.expander("История, которую получит следующая реплика", expanded=True):
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
    if not st.session_state.history:
        st.caption("История пока пустая.")
