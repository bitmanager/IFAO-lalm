"""Two CPU/Gloo ranks: unequal task counts and an empty validation rank."""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lalm'))
from lalm_core.task_loss import separate_task_ce
from lalm_core.trainer import LALMTrainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4, bias=False)
        self.register_buffer('broadcast_probe', torch.ones(1))
        self.audio_tower = torch.nn.Identity()
        self.config = SimpleNamespace(audio_config=SimpleNamespace(model_type='gigaam'))

    def forward(self, input_ids, labels, asr_mask, answer_loss_weight, sync_task_counts, **kwargs):
        logits = self.linear(torch.nn.functional.one_hot(input_ids, 3).float())
        loss, sums, counts = separate_task_ce(logits, labels,
            asr_mask.repeat_interleave(input_ids.shape[1]), answer_loss_weight, sync_task_counts)
        return SimpleNamespace(logits=logits, loss=loss, packed_labels=labels,
            task_nll_sums=sums, task_token_counts=counts)


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(9)
        model = DDP(TinyModel())
        x = torch.tensor([[0, 1, 2, 0]])
        for missing_asr in (False, True):
            labels = torch.tensor([[-100, 1, -100, -100]]) if rank == 0 else torch.tensor([[-100, 2, 1, 0]])
            if missing_asr and rank == 0:
                labels.fill_(-100)
            model.zero_grad(set_to_none=True)
            actual = model(input_ids=x, labels=labels, asr_mask=torch.tensor([rank == 0]),
                answer_loss_weight=2., sync_task_counts=True)
            actual.loss.backward()
            reference = TinyModel()
            reference.load_state_dict(model.module.state_dict())
            target_asr = torch.tensor([[-100, -100, -100, -100] if missing_asr else [-100, 1, -100, -100]])
            ref_asr = reference(input_ids=x, labels=target_asr, asr_mask=torch.tensor([True]),
                answer_loss_weight=2., sync_task_counts=False)
            ref_answer = reference(input_ids=x, labels=torch.tensor([[-100, 2, 1, 0]]),
                asr_mask=torch.tensor([False]), answer_loss_weight=2., sync_task_counts=False)
            (ref_asr.loss + ref_answer.loss).backward()
            torch.testing.assert_close(model.module.linear.weight.grad, reference.linear.weight.grad)
            assert torch.isfinite(actual.loss)

        # Rank0 has no validation batches; rank1 has two. Calling the DDP
        # wrapper during validation would wait for buffer broadcasts and hang.
        trainer = object.__new__(LALMTrainer)
        trainer.model, trainer.device = model, torch.device('cpu')
        trainer.rank, trainer.world_size, trainer.global_step = rank, 2, 0
        trainer.answer_loss_weight, trainer.mixed_precision, trainer.tb_writer = 2., None, None
        batch = dict(input_ids=x, attention_mask=torch.ones_like(x), labels=torch.tensor([[-100, 2, 1, 0]]),
            asr_mask=torch.tensor([False]), features=torch.zeros(1, 1), feature_lens=torch.ones(1, dtype=torch.long))
        trainer.data_module = SimpleNamespace(valid_names=['uneven'], valid_dls=[[] if rank == 0 else [batch, batch]])
        trainer.validate(0)
        assert model.training
        Path(output, f'rank{rank}.done').write_text('PASS')
    finally:
        dist.destroy_process_group()


def test_ddp_task_mean_gradients_and_empty_validation_rank(tmp_path):
    mp.spawn(worker, args=(str(tmp_path/'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    assert all((tmp_path/f'rank{rank}.done').read_text() == 'PASS' for rank in (0, 1))
