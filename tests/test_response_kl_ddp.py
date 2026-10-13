"""CPU/Gloo global response-token mean and unequal validation batch counts."""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lalm'))
from lalm_core.task_loss import response_kl_loss
from lalm_core.trainer import LALMTrainer


class TinyKL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4, bias=False)
        self.register_buffer('broadcast_probe', torch.ones(1))
        self.audio_tower = torch.nn.Identity()
        self.config = SimpleNamespace(audio_config=SimpleNamespace(model_type='gigaam'))

    def forward(self, input_ids, labels, teacher_inputs, sync_task_counts=False, **kwargs):
        logits = self.linear(torch.nn.functional.one_hot(input_ids, 3).float())
        teacher = torch.tensor([1., -.5, .3, .1]).expand_as(logits)
        loss, numerator, count = response_kl_loss(logits, teacher, labels,
            teacher_inputs['labels'], sync_counts=sync_task_counts)
        return SimpleNamespace(logits=logits, loss=loss, packed_labels=labels,
                               response_kl_sum=numerator, response_target_tokens=count)


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(9)
        model = DDP(TinyKL())
        x = torch.tensor([[0, 1, 2, 0]])
        for empty_rank in (False, True):
            targets = [torch.tensor([[-100, 1, -100, -100]]), torch.tensor([[-100, 2, 1, 0]])]
            if empty_rank: targets[0].fill_(-100)
            labels = targets[rank]
            model.zero_grad(set_to_none=True)
            actual = model(input_ids=x, labels=labels, teacher_inputs={'labels':labels}, sync_task_counts=True)
            actual.loss.backward()
            reference = TinyKL(); reference.load_state_dict(model.module.state_dict())
            global_count = sum((t != -100).sum() for t in targets)
            ref = sum(reference(input_ids=x, labels=t, teacher_inputs={'labels':t}).loss
                      * (t != -100).sum() for t in targets) / global_count
            ref.backward()
            torch.testing.assert_close(model.module.linear.weight.grad, reference.linear.weight.grad)
            assert torch.isfinite(actual.loss)

        trainer = object.__new__(LALMTrainer)
        trainer.model, trainer.device = model, torch.device('cpu')
        trainer.rank, trainer.world_size, trainer.global_step = rank, 2, 0
        trainer.answer_loss_weight, trainer.mixed_precision, trainer.tb_writer = None, None, None
        trainer.response_kl, trainer.response_kl_temperature = True, 2.
        labels = torch.tensor([[-100, 2, 1, 0]])
        batch = dict(input_ids=x, attention_mask=torch.ones_like(x), labels=labels,
            teacher_inputs=dict(input_ids=x, attention_mask=torch.ones_like(x), labels=labels),
            asr_mask=torch.tensor([False]), features=torch.zeros(1, 1), feature_lens=torch.ones(1, dtype=torch.long))
        trainer.data_module = SimpleNamespace(valid_names=['uneven'], valid_dls=[[] if rank == 0 else [batch, batch]])
        trainer.validate(0)
        assert model.training
        Path(output, f'rank{rank}.done').write_text('PASS')
    finally:
        dist.destroy_process_group()


def test_response_kl_global_gradients_and_empty_validation_rank(tmp_path):
    mp.spawn(worker, args=(str(tmp_path/'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    assert all((tmp_path/f'rank{rank}.done').read_text() == 'PASS' for rank in (0, 1))
