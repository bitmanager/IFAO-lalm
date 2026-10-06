import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lalm'))
from export_oasst_renderer_scripts import convert


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


@pytest.fixture
def chain():
    root = 'b9d7ade1-38ce-4cb2-ac7c-ce0e282f9b8a'
    ids = [root, 'answer1', 'question2', 'answer2']
    messages = [dict(message_id=mid, parent_id=ids[i-1] if i else None,
                    role='prompter' if i%2==0 else 'assistant', text='  Текст\nбез изменений.  ', rank=None, quality=.8)
                for i,mid in enumerate(ids)]
    selected = dict(root_id=root, source_split='train', conversation_ids=ids, messages=messages, selection_key=[])
    official = {m['message_id']:dict(copy.deepcopy(m), source_split='train', message_tree_id=root,
        labels={'name':['quality'], 'value':[.8]}, lang='ru', deleted=False, synthetic=False, review_result=True) for m in messages}
    return selected, official


def test_original_bytes_and_provenance(chain):
    row, official = chain
    result = convert([row], official, set(), Tokenizer(), {'revision':'fixed'})[0]
    assert [m['text'] for m in result['dialogue']] == [m['text'] for m in row['messages']]
    assert result['source_provenance']['messages'] == row['messages']
    assert [p['role'] for p in result['participants']] == ['user', 'assistant']


@pytest.mark.parametrize('failure', ['validation', 'text', 'parent', 'ids', 'duplicate', 'blocked', 'target_length'])
def test_reject_without_repair(chain, failure):
    row, official = chain
    selected, blocked = [row], set()
    if failure=='validation':
        official[row['root_id']]['source_split']='validation'
    elif failure in ('text','parent'):
        row['messages'][1]['text' if failure=='text' else 'parent_id']='Changed'
    elif failure=='ids':
        row['conversation_ids']=['wrong']
    elif failure=='duplicate':
        selected.append(copy.deepcopy(row))
    elif failure=='blocked':
        blocked.add(row['root_id'])
    else:
        text='word '*257
        row['messages'][1]['text']=official['answer1']['text']=text
    with pytest.raises(ValueError):
        convert(selected, official, blocked, Tokenizer(), {})
