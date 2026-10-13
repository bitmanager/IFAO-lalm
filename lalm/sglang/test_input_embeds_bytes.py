"""Protocol checks for the PR #20207 port; run inside the patched SGLang image."""
import base64
import unittest
from types import SimpleNamespace

import numpy as np
from pydantic import TypeAdapter
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class InputEmbedsBytesTest(unittest.TestCase):
    def test_http_roundtrip_and_batch(self):
        values = np.arange(10240, dtype=np.float32).reshape(2, 5120)
        encoded = base64.b64encode(values.tobytes()).decode()
        adapter = TypeAdapter(GenerateReqInput)
        req = adapter.validate_python(dict(input_embeds_bytes=encoded, input_embeds_shape=[2, 5120]))
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)
        self.assertEqual(req.input_embeds_bytes, values.tobytes())
        owner = SimpleNamespace(max_req_input_len=8192, context_len=12288,
                                model_config=SimpleNamespace(hidden_size=5120))
        decoded = TokenizerManager._decode_input_embeds_bytes(owner, req.input_embeds_bytes, req.input_embeds_shape)
        np.testing.assert_array_equal(decoded, values)
        batch = adapter.validate_python(dict(input_embeds_bytes=[encoded, encoded],
                                            input_embeds_shape=[[2, 5120], [2, 5120]]))
        batch.normalize_batch_and_arguments()
        self.assertFalse(batch.is_single)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch[1].input_embeds_bytes, values.tobytes())

    def test_reject_invalid_input(self):
        for kwargs in (dict(input_embeds_shape=[2, 5120]), dict(input_embeds_bytes=b'x'),
                       dict(text='x', input_ids=[1])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GenerateReqInput(**kwargs).normalize_batch_and_arguments()
        owner = SimpleNamespace(max_req_input_len=8192, context_len=12288,
                                model_config=SimpleNamespace(hidden_size=5120))
        for shape in ([2, 5120], [2, 5119], [0, 5120], [8193, 5120]):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                TokenizerManager._decode_input_embeds_bytes(owner, b'x', shape)


if __name__ == '__main__':
    unittest.main()
