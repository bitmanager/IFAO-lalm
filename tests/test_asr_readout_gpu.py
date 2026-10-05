"""Run explicitly against the warm-start checkpoint and real contextual audio."""
import os
import sys
from pathlib import Path

import torch
from lhotse import CutSet
from lhotse.dataset.input_strategies import AudioSamples

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from lalm_core.data_module import LALMDataset
from lalm_core.model import LALMForConditionalGeneration, LALMProcessor


def test_real_audio_readout_gradient_and_decode():
    directory = os.environ["IFAO_MODEL_PATH"]
    model = LALMForConditionalGeneration.from_pretrained(directory, torch_dtype=torch.bfloat16).cuda()
    processor = LALMProcessor.from_pretrained(directory)
    model.language_model.requires_grad_(False)
    model.audio_tower.requires_grad_(False)
    assert model.config.asr_layer == 35
    torch.testing.assert_close(model.asr_head.weight, model.language_model.lm_head.weight)
    assert model.asr_head.weight.data_ptr() != model.language_model.lm_head.weight.data_ptr()
    cuts = CutSet.from_cuts(list(CutSet.from_file(os.environ["IFAO_MANIFEST_PATH"]))[:2])
    batch = LALMDataset(AudioSamples(fault_tolerant=False), processor)[cuts]
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=batch["input_ids"].cuda(), audio_features=batch["features"].cuda(),
            feature_lens=batch["feature_lens"].cuda(), attention_mask=batch["attention_mask"].cuda(),
            labels=batch["labels"].cuda(), asr_mask=batch["asr_mask"].cuda(),
        )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    expected = {n for n, p in model.named_parameters() if n.startswith(("projector.", "asr_"))}
    assert grads == expected
    assert all(torch.isfinite(p.grad).all() and p.grad.norm() > 0 for n, p in model.named_parameters() if n in expected)
    processor.tokenizer.padding_side = "left"
    asr_cut = next(c for c in cuts if getattr(c, "task", None) == "asr")
    prompt = processor.apply_chat_template(asr_cut.conversation[:-1], add_generation_prompt=True)
    inputs = processor(text=prompt, audio=[asr_cut.load_audio()[0]], sampling_rate=16000,
                       padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated = model.eval().generate(**inputs, asr=True, do_sample=False, max_new_tokens=4)
    assert generated.shape == (1, 4)
    print({"loss": output.loss.item(), "asr_decode": processor.batch_decode(generated),
           "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30})
