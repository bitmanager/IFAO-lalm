"""Response-only BLSP-style KD; native rendering, labels and packed LM are reused."""
import copy

import torch

from prepare_conversation import _render_conversation, teacher_messages


def validate_response_targets(student, teacher, student_mask, teacher_mask):
    """Align by response ordinal per example, never by absolute prefix offset."""
    if student.shape[0] != teacher.shape[0]:
        raise ValueError("Response KL batch sizes differ")
    for labels, mask in ((student, student_mask), (teacher, teacher_mask)):
        if labels.shape != mask.shape or ((labels != -100) & ~mask.bool()).any():
            raise ValueError("Response KL labels must exclude padding")
        for row in labels:
            pos = (row != -100).nonzero().flatten()
            if not len(pos) or pos[0] == 0 or not torch.equal(
                pos, torch.arange(pos[0], pos[-1] + 1, device=pos.device)
            ):
                raise ValueError("Response KL requires one nonempty response after a masked prefix")
    for left, right in zip(student, teacher):
        if not torch.equal(left[left != -100], right[right != -100]):
            raise ValueError("Response KL target token IDs differ within an example")


def prepare_response_teacher(cuts, processor, student):
    texts = []
    for cut in cuts:
        if getattr(cut, "task", "answer") != "answer":
            raise ValueError("Response KL accepts answer-only cuts, not ASR/mixed batches")
        if len(cut.supervisions) != 1:
            raise ValueError("Response KL requires one explicit transcript and answer")
        sup = cut.supervisions[0]
        answer = (sup.custom or {}).get("answer")
        if not isinstance(sup.text, str) or not sup.text.strip() or "<|" in sup.text:
            raise ValueError("Response KL requires a plain foreground transcript")
        if not isinstance(answer, str) or not answer.strip() or "<|" in answer:
            raise ValueError("Response KL requires an explicit plain answer")
        conv = copy.deepcopy(cut.conversation)
        system = conv[0]["content"] if conv and conv[0]["role"] == "system" else None
        if (len(conv) < 2 or conv[:-2] != teacher_messages(cut, system=system)[:-1]
                or conv[-1] != {"role": "assistant", "content": answer}
                or conv[-2]["role"] != "user"):
            raise ValueError("Response KL history/system/answer differs from source metadata")
        if _render_conversation(conv, processor.tokenizer) != cut.rendered_conversation:
            raise ValueError("Response KL student rendering differs from conversation")
        content = conv[-2]["content"]
        if not isinstance(content, list) or sum(p.get("type") == "audio" for p in content) != 1:
            raise ValueError("Response KL requires exactly one current audio item")
        conv[-2]["content"] = [
            {"type": "text", "text": sup.text} if p["type"] == "audio" else p
            for p in content
        ]
        texts.append(_render_conversation(conv, processor.tokenizer))
    teacher = processor(text=texts, prepare_labels=True, return_tensors="pt", padding=True)
    validate_response_targets(student["labels"], teacher["labels"],
                              student["attention_mask"], teacher["attention_mask"])
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for row in teacher["labels"]:
        if row[row != -100][-1].item() != eos:
            raise ValueError("Response KL target must include final assistant EOS")
    return {key: teacher[key] for key in ("input_ids", "attention_mask", "labels")}
