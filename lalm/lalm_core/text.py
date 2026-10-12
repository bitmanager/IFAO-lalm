"""Shared text normalization for ASR training and evaluation."""

import jiwer

ASR_NORMALIZER = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)
