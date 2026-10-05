"""GPU integration check against the real checkpoint, tokenizer and audio cuts."""

import os
import sys
from pathlib import Path

import torch
from lhotse import CutSet
from lhotse.dataset.input_strategies import AudioSamples

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lalm"))
from lalm_core.data_module import LALMDataset
from lalm_core.model import LALMForConditionalGeneration, LALMProcessor


def test_real_audio_gradient_and_placeholder_contract():
    model_dir = os.environ["IFAO_MODEL_PATH"]
    model = LALMForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda()
    processor = LALMProcessor.from_pretrained(model_dir)
    model.audio_tower.requires_grad_(False)
    model.language_model.requires_grad_(False)
    model.train()
    cuts = list(CutSet.from_file(os.environ["IFAO_MANIFEST_PATH"]))
    cuts = CutSet.from_cuts([min(cuts, key=lambda c: c.duration), max(cuts, key=lambda c: c.duration)])
    dataset = LALMDataset(AudioSamples(fault_tolerant=False), processor)
    batch = dataset[cuts]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=batch["input_ids"].cuda(),
            audio_features=batch["features"].cuda(),
            feature_lens=batch["feature_lens"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            labels=batch["labels"].cuda(),
        )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    trained = {name for name, p in model.named_parameters() if p.grad is not None}
    assert trained == {f"projector.{name}" for name, _ in model.projector.named_parameters()}
    assert all(torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in model.projector.parameters())
    assert not model.audio_tower.model.training
    # Exercise the production raw-audio processor and the upstream generation path,
    # which runs without the trainer's autocast context.
    processor.tokenizer.padding_side = "left"
    prompts = [processor.apply_chat_template(c.conversation[:-1], add_generation_prompt=True) for c in cuts]
    inputs = processor(text=prompts, audio=[c.load_audio()[0] for c in cuts],
                       sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")
    assert inputs.input_features.dtype == torch.float32
    with torch.no_grad():
        generated = model.eval().generate(**inputs, do_sample=False, max_new_tokens=2)
    assert generated.size(0) == 2
    print({"loss": output.loss.item(), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "llm_dtype": str(next(model.language_model.parameters()).dtype),
           "attention": model.language_model.config._attn_implementation})
