from __future__ import annotations

import logging

import torch
import yaml
from lhotse import CutSet, set_audio_duration_mismatch_tolerance
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.sampling.base import TokenConstraint
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data import DataLoader

from auden.data.lhotse_datamodule import BaseLhotseDatamodule, _SeedWorkers
from .paired_tasks import expand_task_views, task_views


def estimate_cut_tokens(cut, audio_token_rate: float, paired_tasks: bool = False):
    """Estimate total tokens from duration and prepared num_text_tokens."""
    num_text_tokens = getattr(cut, "num_text_tokens", None)
    if num_text_tokens is None:
        raise ValueError(
            f"Cut {getattr(cut, 'id', '<unknown>')} missing cut.num_text_tokens. "
            "Please run prepare_conversation.py first."
        )

    views = task_views(cut)
    if views is not None:
        if not paired_tasks:
            raise ValueError("Task-view manifests require data.paired_tasks=true")
        num_text_tokens = sum(v["num_text_tokens"] for v in views)
    num_audio_tokens = int(round(float(cut.duration) * float(audio_token_rate)))
    num_audio_tokens *= len(views) if views is not None else 1
    num_tokens = num_audio_tokens + int(num_text_tokens)
    cut.num_tokens = num_tokens
    return cut


class LALMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_strategy,
        processor,
        cut_transforms=None,
        input_transforms=None,
        return_cuts: bool = False,
        paired_tasks: bool = False,
    ):
        self.input_strategy = input_strategy
        self.processor = processor
        self.cut_transforms = cut_transforms
        self.input_transforms = input_transforms
        self.return_cuts = return_cuts
        self.paired_tasks = paired_tasks
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts) -> dict:
        self.hdf5_fix.update()

        if self.cut_transforms is not None:
            for transform in self.cut_transforms:
                cuts = transform(cuts)
        if self.paired_tasks:
            cuts = expand_task_views(cuts)
        elif any(task_views(c) is not None for c in cuts):
            raise ValueError("Task-view manifests require data.paired_tasks=true")
        cuts = cuts.sort_by_duration(ascending=False)
        expected_ids = set(cuts.ids)

        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            features, _, cuts = input_tpl
        else:
            features, _ = input_tpl
        if self.paired_tasks and set(cuts.ids) != expected_ids:
            raise ValueError("Input loading dropped a linked task view")

        supervision_intervals = self.input_strategy.supervision_intervals(cuts)
        feature_lens = supervision_intervals[
            "num_samples" if self.processor.encoder_name == "gigaam" else "num_frames"
        ]
        feature_lens = feature_lens.to(dtype=torch.long)

        # lhotse returns (N, T, C); processor expects padded (N, C, T_max)
        if features.ndim == 3:
            features = features.transpose(1, 2).contiguous()

        rendered_texts = []
        for cut in cuts:
            if getattr(cut, "task", "answer") not in ("answer", "asr"):
                raise ValueError(f"Cut {cut.id}: unsupported training task")
            rendered_text = getattr(cut, "rendered_conversation", None)
            if rendered_text is None:
                raise ValueError(
                    f"Cut {getattr(cut, 'id', '<unknown>')} missing "
                    "cut.rendered_conversation. Please run prepare_conversation.py first."
                )
            rendered_texts.append(rendered_text)

        inputs = self.processor(
            text=rendered_texts,
            audio_feature=(features, feature_lens),
            prepare_labels=True,
            return_tensors="pt",
            padding=True,
        )

        batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "features": inputs["input_features"],
            "feature_lens": inputs["feature_lens"],
            "labels": inputs["labels"],
            "batch_size": inputs["input_ids"].size(0),
            "asr_mask": torch.tensor([getattr(c, "task", "answer") == "asr" for c in cuts]),
        }

        flat_cuts = [cut for cut in cuts for _ in cut.supervisions]
        if self.return_cuts:
            batch["cuts"] = flat_cuts
        return batch

    def __len__(self) -> int:
        return 0


class LALMDataModule(BaseLhotseDatamodule):
    def __init__(self, cfg, processor):
        set_audio_duration_mismatch_tolerance(1)
        self.processor = processor
        super().__init__(cfg)

    def _filter_cutset(self, cutset: CutSet, split: str = "train") -> CutSet:
        min_dur = float(self.cfg.get("min_duration", 0.5))
        max_dur = float(self.cfg.get("max_duration", 30.0))

        def keep(c):
            return min_dur <= c.duration <= max_dur

        audio_token_rate = float(self.cfg.get("audio_token_rate", 12.5))
        return cutset.filter(keep).map(
            lambda cut: estimate_cut_tokens(cut, audio_token_rate=audio_token_rate,
                paired_tasks=self.cfg.get("paired_tasks", False))
        )

    def setup_train(self):
        with open(self.cfg.train_data_config, "r", encoding="utf-8") as f:
            train_data_config = yaml.load(f, Loader=yaml.FullLoader)

        train_cutset = self._build_train_mux_cutset(train_data_config)
        train_cutset = self._filter_cutset(train_cutset, split="train")

        max_tokens = self.cfg.sampler.get("max_tokens", None)
        max_duration = self.cfg.sampler.get("max_duration", None)
        if self.cfg.get("paired_tasks", False) and max_tokens is None:
            raise ValueError("Linked task views require a summed max_tokens budget")
        num_buckets = self.cfg.sampler.get("num_buckets", 30)
        common = dict(
            shuffle=self.cfg.sampler.shuffle,
            num_buckets=num_buckets,
            buffer_size=num_buckets * 2000,
            shuffle_buffer_size=num_buckets * 5000,
            drop_last=self.cfg.sampler.get("drop_last", True),
        )

        if max_tokens is not None:
            logging.info(
                f"[data] Train sampler: TokenConstraint max_tokens={max_tokens}"
            )
            train_sampler = DynamicBucketingSampler(
                train_cutset,
                constraint=TokenConstraint(max_tokens=max_tokens),
                **common,
            )
        elif max_duration is not None:
            logging.info(f"[data] Train sampler: max_duration={max_duration}s")
            train_sampler = DynamicBucketingSampler(
                train_cutset, max_duration=max_duration, **common
            )
        else:
            raise ValueError(
                "sampler must set max_tokens (recommended) or max_duration"
            )

        train_dataset = LALMDataset(
            input_strategy=self.input_strategy,
            processor=self.processor,
            cut_transforms=self.transforms,
            input_transforms=self.input_transforms,
            return_cuts=True,
            paired_tasks=self.cfg.get("paired_tasks", False),
        )
        seed = torch.randint(0, 100_000, ()).item()
        worker_init_fn = _SeedWorkers(seed)
        self.train_dl = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.cfg.get("num_workers", 4),
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    def setup_valid(self):
        valid_cfg_path = self.cfg.get("valid_data_config", None)
        if not valid_cfg_path:
            self.valid_dls = []
            self.valid_names = []
            return

        with open(valid_cfg_path, "r", encoding="utf-8") as f:
            valid_data_config = yaml.load(f, Loader=yaml.FullLoader)

        self.valid_dls = []
        self.valid_names = []
        for valid_set in valid_data_config:
            cutset = CutSet.from_file(valid_set["manifest"]).resample(
                self.sampling_rate
            )
            cutset = self._filter_cutset(cutset, split="valid")
            valid_name = valid_set.get("name", "valid")

            max_tokens = self.cfg.sampler.get("max_tokens", None)
            max_duration = self.cfg.sampler.get("max_duration", None)
            if max_tokens is not None:
                valid_sampler = DynamicBucketingSampler(
                    cutset,
                    constraint=TokenConstraint(max_tokens=max_tokens),
                    num_buckets=self.cfg.sampler.get("num_buckets", 30),
                    shuffle=False,
                )
            else:
                valid_sampler = DynamicBucketingSampler(
                    cutset, max_duration=max_duration,
                    num_buckets=self.cfg.sampler.get("num_buckets", 30), shuffle=False
                )

            valid_dataset = LALMDataset(
                input_strategy=self.input_strategy,
                processor=self.processor,
                return_cuts=True,
                paired_tasks=self.cfg.get("paired_tasks", False),
            )
            valid_dl = DataLoader(
                valid_dataset,
                sampler=valid_sampler,
                batch_size=None,
                num_workers=self.cfg.get("num_workers", 4),
                persistent_workers=False,
            )
            self.valid_names.append(valid_name)
            self.valid_dls.append(valid_dl)
