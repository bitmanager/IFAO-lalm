import unittest

from lalm.benchmark_qwen_calls import inference_messages, context_copy_flag
from lalm.prepare_qwen_history import group_speech_segments


class HistoryV2Tests(unittest.TestCase):
    def test_v3_keeps_context_before_audio_and_diagnostic_control_omits_it(self):
        history = [{"role": "user", "content": "старая реплика"}]
        user = {"role": "user", "content": "<|audio|>"}
        messages = inference_messages("ASR", history, user, "reference_history_v3")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertTrue(messages[-1]["content"].endswith("<|audio|>"))
        self.assertIn("Конец справки", messages[-1]["content"])
        self.assertEqual(user["content"], "<|audio|>")
        control = inference_messages("ASR", history, user, "no_history")
        self.assertNotIn("старая реплика", str(control))

    def test_copy_warning_does_not_flag_short_agreements_or_modify_text(self):
        history = [{"role": "user", "content": "Маркетинг и аналитика в одной роли."}]
        self.assertTrue(context_copy_flag("маркетинг и аналитика в одной роли", history))
        self.assertFalse(context_copy_flag("Да", [{"content": "Да"}]))
        self.assertFalse(context_copy_flag("Мне двадцать один", history))

    def test_history_is_not_rendered_as_active_chat_turns(self):
        history = [{"role": "user", "content": "старый текст"},
                   {"role": "assistant", "content": "старый ответ"}]
        user = {"role": "user", "content": "<|audio|> текущий звук"}
        messages = inference_messages("Расшифруй", history, user, "reference_history_v2")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[-1], user)
        self.assertIn("старый текст", messages[0]["content"])
        self.assertIn("старый ответ", messages[0]["content"])

    def test_merge_preserves_coverage_and_agent_boundaries(self):
        def s(i, a, b, h):
            return dict(id=str(i), start=a, end=b, native_vad_start=a,
                        native_vad_end=b, history=h)
        history = [{"role": "assistant", "content": "новый вопрос", "source_at_seconds": 3}]
        source = [s(0, 0, 1, []), s(1, 1.3, 2.8, []), s(2, 3.1, 4, history),
                  s(3, 4.2, 29, history), s(4, 29.1, 34, history)]
        result = group_speech_segments(source)
        self.assertEqual([(s["start"], s["end"]) for s in result], [(0, 2.8), (3.1, 29), (29.1, 34)])
        self.assertEqual(result[0]["history"], [])
        self.assertEqual(result[0]["source_segment_ids"], ["0", "1"])
        self.assertEqual(source[0]["end"], 1)


if __name__ == "__main__":
    unittest.main()
