import json
from pathlib import Path
import pytest
from neurosleepnet.embeddings.onnx_pooled import ONNXPooledEncoder
from neurosleepnet.embeddings.pooled import AssetNotFoundError


def assets(path):
    (path / '1_Pooling').mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps({'model_type': 'bert', 'hidden_size': 384}))
    (path / '1_Pooling/config.json').write_text(json.dumps({'pooling_mode_mean_tokens': True}))
    (path / 'sentence_bert_config.json').write_text(json.dumps({'max_seq_length': 256}))
    (path / 'tokenizer.json').write_text('{}')
    (path / 'model.onnx').write_bytes(b'fixture: loading this must fail')
    return path


def test_onnx_offline_missing_assets_and_lazy_empty_batch(tmp_path):
    with pytest.raises(AssetNotFoundError):
        ONNXPooledEncoder(tmp_path)
    encoder = ONNXPooledEncoder(assets(tmp_path))
    assert encoder.encode_batch([]) == []
    assert encoder._session is None


def test_onnx_changed_assets_cannot_reopen_old_vectors(tmp_path):
    from nsn import Runtime
    from neurosleepnet.storage.compact_vector import IncompatibleIndexError
    path = assets(tmp_path / 'assets')
    original = ONNXPooledEncoder(path)
    rt = Runtime(tmp_path / 'data', encoder=original)
    rt.close()
    (path / 'tokenizer.json').write_text('{"changed": true}')
    with pytest.raises(IncompatibleIndexError):
        Runtime(tmp_path / 'data', encoder=ONNXPooledEncoder(path))


def test_onnx_rejects_different_pooling_semantics(tmp_path):
    path = assets(tmp_path)
    (path / '1_Pooling/config.json').write_text(json.dumps({'pooling_mode_mean_tokens': True, 'pooling_mode_cls_token': True}))
    with pytest.raises(ValueError, match='mean-pooling'):
        ONNXPooledEncoder(path)


def test_large_haystack_bounds_inference_allocations_and_keeps_order(tmp_path, monkeypatch):
    encoder = ONNXPooledEncoder(assets(tmp_path))
    calls = []
    monkeypatch.setattr(encoder, '_load', lambda: None)
    def chunk(texts):
        calls.append(len(texts))
        return [[int(text)] for text in texts]
    monkeypatch.setattr(encoder, '_encode_chunk', chunk)
    assert encoder.encode_batch([str(i) for i in range(130)]) == [[i] for i in range(130)]
    assert calls == [64, 64, 2]


def test_real_onnx_offline_restart_and_identifier_recall(tmp_path, monkeypatch):
    import os
    import socket
    asset_path = Path(os.environ.get('NSN_ONNX_ASSETS', 'C:/Users/aviroop/AppData/Local/NSN/evaluation/minilm-onnx-1110a243'))
    if not asset_path.is_dir():
        pytest.skip('Explicitly prepared ONNX assets unavailable')
    pytest.importorskip('onnxruntime')
    from nsn import Runtime
    def denied(*a, **k):
        raise AssertionError('Unexpected network request')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    encoder = ONNXPooledEncoder(asset_path, cache_size=0)
    with Runtime(tmp_path, encoder=encoder) as rt:
        eid = rt.append_event(namespace='agent', content='Payment gateway production port is 8080.')
    with Runtime(tmp_path, encoder=encoder) as rt:
        pack = rt.retrieve_pack('Where does the payment gateway listen?', namespace='agent')
        assert eid in {item.id for item in pack.items}
        assert '8080' in pack.rendered_text
        assert not rt.retrieve_pack('Payment gateway', namespace='other').items
        rt.storage.delete_event(eid, 'agent')
        assert not rt.retrieve_pack('Payment gateway', namespace='agent').items
