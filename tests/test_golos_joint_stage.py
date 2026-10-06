"""CPU-only native sampler coverage and full-QA gates; no model initialization."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lalm'))
from lhotse import AudioSource, CutSet, MonoCut, Recording

spec = importlib.util.spec_from_file_location('golos_joint_review',
    Path(__file__).resolve().parents[1] / 'lalm/scripts/prepare_golos_joint_stage.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
audit_sampler, check_full_qa = module.audit_sampler, module.check_full_qa


def full_qa():
    return dict(status='CPU_QA_PASS_STAGING_ONLY', original_farfield_rows=124003,
        streamed_source_rows=124003, unique_source_ids=124003, original_mirror_text_matches=124003,
        full_original_id_set_equal=True, missing_ids=[], extra_ids=[], official_test_id_overlap=[],
        native_heldout_id_file_pcm_intersections=0, native_processor_batch16_labels_eos_pass=True,
        all_native_texts_roles_paths_checked=True,
        unique_native_pcm_hours=124.9)


@pytest.mark.parametrize('key,value', [
    ('streamed_source_rows', 1000), ('official_test_id_overlap', ['test-id']),
    ('native_heldout_id_file_pcm_intersections', 1), ('unique_native_pcm_hours', 99.9),
    ('native_processor_batch16_labels_eos_pass', False),
    ('all_native_texts_roles_paths_checked', False),
])
def test_reject_partial_or_unqualified_qa(key, value):
    qa = full_qa()
    check_full_qa(qa)
    qa[key] = value
    with pytest.raises(AssertionError):
        check_full_qa(qa)


def fixture_cuts():
    cuts = []
    for i in range(40):
        custom = dict(task='asr' if i < 30 else 'answer', num_text_tokens=80)
        if i < 30:
            custom.update(original_id=f'g{i}', pcm_sha256=f'pcm{max(1, i)}')
        recording = Recording(id=f'r{i}', sampling_rate=16000, num_samples=160000,
            duration=10, sources=[AudioSource(type='file', channels=[0], source='/unused.wav')])
        cuts.append(MonoCut(id=f'cut{i}', start=0, duration=10, channel=0,
            recording=recording, custom=custom))
    return CutSet.from_cuts(cuts)


def test_native_two_rank_unique_pcm_coverage_and_determinism():
    cuts = fixture_cuts()
    expected = {f'g{i}' for i in range(30)}
    result = audit_sampler(copy.deepcopy(cuts), expected, target_hours=.06)
    assert result == audit_sampler(copy.deepcopy(cuts), expected, target_hours=.06)
    assert result['batches_per_rank']['0'] == result['batches_per_rank']['1']
    assert result['num_steps'] == result['updates'] - 1
    assert 0 < result['N100'] <= result['updates']
    assert result['unique_golos_source_ids'] == 30
    assert result['unique_golos_pcm'] == 29
    assert result['unique_golos_pcm_hours'] == pytest.approx(290 / 3600)
    assert result['full_native_view_id_coverage']


def test_native_sampler_rejects_unattainable_unique_hours():
    with pytest.raises(AssertionError, match='did not reach'):
        audit_sampler(fixture_cuts(), {f'g{i}' for i in range(30)}, target_hours=1)


def test_token_metadata_reconciliation_preserves_every_other_field():
    from types import SimpleNamespace
    cut = next(iter(fixture_cuts()))
    cut.custom.update(num_text_tokens=6, rendered_conversation=' <|audio|> text\n')
    before = copy.deepcopy(cut.to_dict())
    class Tokenizer:
        def __init__(self, audio_ids): self.audio_ids = audio_ids
        def __call__(self, text, add_special_tokens):
            assert text == '<|audio|> text' and add_special_tokens is False
            return SimpleNamespace(input_ids=self.audio_ids + [1495])
    source, native = Tokenizer([27, 91, 16736, 91, 29]), Tokenizer([151669])
    assert module.reconcile_golos_token_count(cut, source, native) == (6, 2)
    before['custom']['num_text_tokens'] = 2
    assert cut.to_dict() == before
    with pytest.raises(AssertionError):
        module.reconcile_golos_token_count(cut, source, native)


@pytest.mark.parametrize('bad_field,bad_value', [
    ('status', 'pending'), ('test_decoded_rows', 1915),
    ('train_manifest_sha256', 'stale'),
    ('intersections', dict(source_ids=0, encoded_sha256=0, pcm_sha256=1)),
])
def test_official_test_gate_rejects_pending_partial_stale_or_overlap(bad_field, bad_value):
    qa = dict(status='PASS', train_manifest_sha256='train', train_source_index_sha256='index',
        train_source_rows=124003, test_decoded_rows=1916, test_native_cuts=1915, test_unresolved_rows=1,
        intersections=dict(source_ids=0, encoded_sha256=0, pcm_sha256=0),
        test_source_index_path=str(module.TEST / 'source-index.jsonl'),
        test_source_index_sha256='test-index', test_manifest_sha256='test-manifest')
    assert len(module.check_test_overlap(qa, 'train', 'index')) == 2
    qa[bad_field] = bad_value
    with pytest.raises(AssertionError):
        module.check_test_overlap(qa, 'train', 'index')


def test_replay_only_accepts_verified_conservative_legacy_count():
    from types import SimpleNamespace
    cut = next(iter(fixture_cuts()))
    cut.custom.update(num_text_tokens=73, rendered_conversation=' exact text ')
    source = lambda text, add_special_tokens: SimpleNamespace(input_ids=list(range(73)))
    before = copy.deepcopy(cut.to_dict())
    assert module.validate_replay_token_count(cut, source, 69) == 4
    assert cut.to_dict() == before
    cut.custom['num_text_tokens'] = 69
    assert module.validate_replay_token_count(cut, source, 69) == 0
    for bad in (68, 72, 74):
        cut.custom['num_text_tokens'] = bad
        with pytest.raises(AssertionError):
            module.validate_replay_token_count(cut, source, 69)
