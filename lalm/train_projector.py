"""Portable response-KL projector recipe; optimization stays in LALMTrainer."""

import copy
import logging
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from lalm_core import LALMDataModule, LALMTrainer
from lalm_core.model import LALMConfig, LALMForConditionalGeneration
from lalm_core.model.gigaam_adapter import GigaAMAudioTower
from lalm_core.model.loading import load_language_model
from lhotse.utils import fix_random_seed
from omegaconf import OmegaConf
from projector_runtime import install_eval_checkpointing, processor
from torch.nn.parallel import DistributedDataParallel as DDP


class ProjectorTrainer(LALMTrainer):
    """Avoid copying/saving the frozen 27B backbone for EMA and checkpoints."""

    def setup_model(self, model):
        model.validate_response_kl_freeze()
        self.projector = model.projector
        average = torch.nn.Module()
        average.projector = copy.deepcopy(model.projector).float().cpu()
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)
        return model, average if self.rank == 0 else None

    def _maybe_update_model_average(self):
        if (
            self.rank == 0
            and self.global_step
            and self.global_step % self.cfg.trainer.average_period == 0
        ):
            with torch.no_grad():
                for average, current in zip(
                    self.model_avg.projector.parameters(), self.projector.parameters()
                ):
                    average.mul_(self._ema_decay).add_(
                        current.detach().float().cpu(), alpha=1 - self._ema_decay
                    )

    def save_projector(self, name, epoch):
        if self.rank:
            return
        path = Path(self.cfg.exp_dir) / name
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "projector": self.projector.state_dict(),
                "projector_ema": self.model_avg.projector.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.global_step,
                "epoch": epoch,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
            temporary,
        )
        temporary.replace(path)

    def _maybe_validate_and_save(self, epoch):
        interval = self.cfg.trainer.valid_interval
        if interval > 0 and self.global_step and self.global_step % interval == 0:
            self.validate(epoch)
        interval = self.cfg.trainer.checkpoint_interval
        if interval > 0 and self.global_step and self.global_step % interval == 0:
            self.save_projector(f"checkpoint-{self.global_step}.pt", epoch)

    def _maybe_save_epoch_checkpoint(self, epoch):
        self.save_projector(f"epoch-{epoch}.pt", epoch)


@hydra.main(version_base=None, config_path="configs", config_name="train_projector_27b")
def main(cfg):
    if cfg.trainer.start_batch or cfg.trainer.start_epoch != 1:
        raise ValueError(
            "Compact checkpoints support projector initialization, not exact resume"
        )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        # Refuse accidental reuse of an experiment directory, including on nonzero ranks.
        if rank == 0:
            Path(cfg.exp_dir).mkdir(parents=True, exist_ok=True)
            with (Path(cfg.exp_dir) / "recipe.yaml").open("x") as output:
                output.write(OmegaConf.to_yaml(cfg, resolve=True))
        if world > 1:
            dist.barrier()
        fix_random_seed(cfg.seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        device = f"cuda:{local_rank}"
        lm = (
            load_language_model(
                cfg.model.language_model,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                attn_implementation="sdpa",
            )
            .requires_grad_(False)
            .eval()
        )
        lm.config.use_cache = False
        if cfg.model.get("gemma_model"):
            from lalm_core.model.gemma_adapter import GemmaAudioTower

            if cfg.model.initial_projector:
                raise ValueError("Gemma pilot requires a fresh projector")
            encoder = GemmaAudioTower.from_pretrained(cfg.model.gemma_model).to(device)
            audio_dim = encoder.audio_dim
        else:
            encoder = GigaAMAudioTower.from_checkpoint(cfg.model.gigaam_checkpoint)
            audio_dim = encoder.model.cfg.encoder.d_model
        encoder.requires_grad_(False).eval()
        proc = processor(cfg.model.tokenizer, cfg.model.projector_downsample_rate, encoder.config.model_type)
        if (
            proc.tokenizer.convert_tokens_to_ids("<|audio|>")
            >= lm.get_input_embeddings().num_embeddings
        ):
            raise ValueError("Audio placeholder exceeds the frozen model vocabulary")
        model = LALMForConditionalGeneration(
            LALMConfig(
                text_config=lm.config,
                audio_config=encoder.config,
                audio_dim=audio_dim,
                text_dim=lm.config.hidden_size,
                projector_downsample_rate=cfg.model.projector_downsample_rate,
                projector_hidden_size=cfg.model.projector_hidden_size,
                audio_token_id=proc.tokenizer.convert_tokens_to_ids("<|audio|>"),
            ),
            language_model=lm,
            audio_tower=encoder,
        )
        if cfg.model.projector_downsample_rate != proc.projector_downsample_rate:
            raise ValueError("Processor and projector downsampling must match")
        model.projector.to(device, dtype=torch.float32)
        if cfg.model.initial_projector:
            saved = torch.load(
                cfg.model.initial_projector, map_location="cpu", weights_only=True
            )
            model.projector.load_state_dict(saved["projector"])
        install_eval_checkpointing(lm)
        logging.basicConfig(level=logging.INFO)
        trainer = ProjectorTrainer(
            cfg, model, LALMDataModule(cfg.data, proc), rank, local_rank, world
        )
        trainer.validate(0)
        trainer.run()
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
