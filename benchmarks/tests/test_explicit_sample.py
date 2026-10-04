import pytest
from benchmarks.public_eval.ledger import select_explicit


def test_explicit_sample_preserves_native_order_without_selecting_by_answers():
    cases=[{'id':'a','answer':'wrong'},{'id':'b','answer':'right'},{'id':'c'}]
    assert select_explicit(cases,['c','a'])==[cases[0],cases[2]]
    assert cases==[{'id':'a','answer':'wrong'},{'id':'b','answer':'right'},{'id':'c'}]


@pytest.mark.parametrize('ids',[[],['a','a'],['missing']])
def test_explicit_sample_rejects_invalid_identities(ids):
    with pytest.raises(ValueError,match='explicit question selection'):
        select_explicit([{'id':'a'}],ids)
