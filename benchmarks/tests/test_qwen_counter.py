import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from benchmarks.competitor_eval.qwen_counter import QwenChatCounter


def test_exact_counter_rejects_unverified_request_shapes():
    for messages in ([{'role':'user','content':'question'}],
                     [{'role':'system','content':'system'},{'role':'user','content':'question','images':['image']}],
                     [{'role':'system','content':'system'},{'role':'user','content':['multimodal']}],
                     [{'role':'system','content':'system'},{'role':'assistant','content':'answer'}]):
        with pytest.raises(ValueError,match='only plain'):
            QwenChatCounter.render(messages)


def test_counter_identity_must_match_actual_model(tmp_path):
    (tmp_path/'identity.json').write_text(json.dumps({'model_digest':'pinned'}))
    with pytest.raises(ValueError,match='model digest'):
        QwenChatCounter(tmp_path,SimpleNamespace(identity={'digest':'different'}))


def test_prepared_counter_replays_measured_server_counts_offline():
    root=Path('C:/Users/aviroop/AppData/Local/NSN/evaluation/qwen3-tokenizer')
    qualification=Path(__file__).resolve().parents[2]/'docs/implementation/results/09-qwen-chat-counter-qualification-20261004.json'
    if not root.is_dir() or not qualification.is_file():
        pytest.skip('Explicit counter assets/actual server qualification unavailable')
    show=json.loads((root/'show.json').read_text(encoding='utf-8'))
    artifact=json.loads(qualification.read_text(encoding='utf-8'))
    counter=QwenChatCounter(root,SimpleNamespace(identity={'digest':artifact['counter_identity']['model_digest']},show=show))
    assert artifact['passed']
    for row in artifact['qualification']:
        assert counter.count(row['messages'])==row['observed_prompt_eval_count']
