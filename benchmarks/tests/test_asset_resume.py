import hashlib
import io
import json
import pytest
from benchmarks.public_eval.prepare import prepare


@pytest.mark.parametrize('status', [200, 206])
def test_download_resume_checks_entire_asset_and_handles_ignored_range(tmp_path, monkeypatch, status):
    data=b'pinned native haystack';name='data.json'
    pins=tmp_path/'pins.json'
    pins.write_text(json.dumps({'files':{name:{'url':'https://example.test/data','size':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}),encoding='utf-8')
    root=tmp_path/'assets';root.mkdir();(root/(name+'.part')).write_bytes(data[:6])
    def response(request,timeout):
        assert request.get_header('Range')=='bytes=6-'
        result=io.BytesIO(data[6:] if status==206 else data)
        result.status=status;result.headers={'Content-Range':f'bytes 6-{len(data)-1}/{len(data)}'}
        return result
    monkeypatch.setattr('urllib.request.urlopen',response)
    prepare(root,pins)
    assert (root/name).read_bytes()==data
    assert not (root/(name+'.part')).exists()
    assert name in json.loads((root/'sources.json').read_text(encoding='utf-8'))['files']


def test_wrong_range_or_checksum_never_promotes_partial_asset(tmp_path, monkeypatch):
    root=tmp_path/'assets';root.mkdir();(root/'data.json.part').write_bytes(b'abc')
    pins=tmp_path/'pins.json'
    pins.write_text(json.dumps({'files':{'data.json':{'url':'https://example.test/data','size':6,'sha256':hashlib.sha256(b'abcdef').hexdigest()}}}),encoding='utf-8')
    def response(request,timeout):
        result=io.BytesIO(b'def');result.status=206;result.headers={'Content-Range':'bytes 2-5/6'};return result
    monkeypatch.setattr('urllib.request.urlopen',response)
    with pytest.raises(ValueError,match='Invalid range'):prepare(root,pins)
    assert (root/'data.json.part').read_bytes()==b'abc' and not (root/'data.json').exists()
    def corrupt(request,timeout):
        result=io.BytesIO(b'xxx');result.status=206;result.headers={'Content-Range':'bytes 3-5/6'};return result
    monkeypatch.setattr('urllib.request.urlopen',corrupt)
    with pytest.raises(ValueError,match='Pinned source'):prepare(root,pins)
    assert not (root/'data.json').exists()
