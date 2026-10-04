import json
from types import SimpleNamespace
import pytest
from benchmarks.public_eval.mem0_public import source_revision


def test_actual_competitor_source_pin_is_required_without_version_substitution():
    for url in (None,json.dumps({'vcs_info':{'commit_id':'different'}})):
        with pytest.raises(ValueError,match='pin'):source_revision(SimpleNamespace(read_text=lambda name:url))
    pin='abb81c88e1f738a8117d8293530fbc31a5ef8fd9'
    assert source_revision(SimpleNamespace(read_text=lambda name:json.dumps({'vcs_info':{'commit_id':pin}})))==pin
