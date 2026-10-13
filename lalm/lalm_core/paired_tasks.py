"""Atomic sampler units with explicitly labelled, linked training views."""
import copy

from lhotse import CutSet


def task_views(cut):
    views = (cut.custom or {}).get("task_views")
    if views is None:
        return None
    if not 1 <= len(views) <= 2 or len({v["task"] for v in views}) != len(views):
        raise ValueError(f"{cut.id}: expected one or two distinct task views")
    for view in views:
        if view["task"] not in ("asr", "answer"):
            raise ValueError(f"{cut.id}: unsupported paired task")
        if not isinstance(view["target"], str) or not view["target"].strip():
            raise ValueError(f"{cut.id}: a missing target must not create a view")
        if view["conversation"][-1] != {"role": "assistant", "content": view["target"]}:
            raise ValueError(f"{cut.id}: view target and conversation disagree")
        if view["task"] == "asr" and view["target"] != cut.supervisions[0].text:
            raise ValueError(f"{cut.id}: ASR target differs from the explicit transcript")
        if view["num_text_tokens"] <= 0:
            raise ValueError(f"{cut.id}: invalid view token count")
    return views


def expand_task_views(cuts):
    """Expand after sampling/transforms, so a pair cannot straddle ranks/batches."""
    expanded = []
    for cut in cuts:
        views = task_views(cut)
        if views is None:
            expanded.append(cut)
            continue
        for view in views:
            item = copy.deepcopy(cut)
            item.id = f"{cut.id}-{view['task']}"
            item.custom.pop("task_views")
            item.custom.update({k: view[k] for k in (
                "task", "conversation", "rendered_conversation", "num_text_tokens")})
            item.custom["task_pair_id"] = cut.id
            item.supervisions[0].custom = {
                **(item.supervisions[0].custom or {}), "answer": view["target"]}
            expanded.append(item)
    if len({c.id for c in expanded}) != len(expanded):
        raise ValueError("Duplicate expanded task-view IDs")
    return CutSet.from_cuts(expanded)
