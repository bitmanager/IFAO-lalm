"""Convert existing single-utterance Lhotse ASR cuts to the auxiliary Qwen task."""
import argparse
import copy
import json
from pathlib import Path

from lhotse import CutSet, MonoCut
from lhotse.cut import MixedCut, PaddingCut
from transformers import AutoTokenizer

from prepare_conversation import prepare_asr_cut


def asr_cut(source, tokenizer, system):
    if isinstance(source, MixedCut):
        speech = [t for t in source.tracks if not isinstance(t.cut, PaddingCut)]
        if len(speech) != 1 or speech[0].snr is not None:
            raise ValueError(f"{source.id}: expected one unmodified audio cut plus padding")
        source = speech[0].cut
    if not isinstance(source, MonoCut) or len(source.supervisions) != 1:
        raise ValueError(f"{source.id}: expected one mono utterance and transcript")
    cut = copy.deepcopy(source)
    sup = cut.supervisions[0]
    if abs(sup.start) > 0.001 or abs(sup.duration - cut.duration) > 0.001:
        raise ValueError(f"{cut.id}: transcript must cover the entire unpadded cut")
    if sup.language not in ("ru", "Russian") or not sup.text or "<|" in sup.text:
        raise ValueError(f"{cut.id}: expected plain Russian transcript")
    if (cut.custom or {}).get("history"):
        raise ValueError(f"{cut.id}: use contextual preparation for dialogue histories")
    for audio in cut.recording.sources:
        if audio.type != "file" or not Path(audio.source).is_file():
            raise ValueError(f"{cut.id}: expected existing local audio")
    cut.custom = {**(cut.custom or {}), "system": system}
    return prepare_asr_cut(cut, tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    system = args.system_file.read_text().strip()
    if not system or "<|" in system:
        raise ValueError("Expected nonempty plain system prompt")
    seen, seconds = set(), 0.0
    with CutSet.open_writer(args.output_manifest, overwrite=False) as writer:
        for source in CutSet.from_file(args.input_manifest):
            cut = asr_cut(source, tokenizer, system)
            if cut.id in seen:
                raise ValueError(f"Duplicate cut: {cut.id}")
            if not 0.5 <= cut.duration <= 30:
                raise ValueError(f"{cut.id}: cut is outside the training duration range")
            seen.add(cut.id)
            seconds += cut.duration
            writer.write(cut)
    print(json.dumps({"cuts": len(seen), "hours": seconds/3600, "task": "asr", "history": False}))


if __name__ == "__main__":
    main()
