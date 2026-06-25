import pytest
from src.betting import recommend, expected_values


def test_recommend_high_confidence_single_outcome():
    rec = recommend({'H': 0.65, 'D': 0.20, 'A': 0.15})
    assert rec['market'] == 'H'
    assert rec['confidence'] == 'high'


def test_recommend_medium_confidence_single_outcome():
    rec = recommend({'H': 0.15, 'D': 0.30, 'A': 0.55})
    assert rec['market'] == 'A'
    assert rec['confidence'] == 'medium'


def test_recommend_double_chance():
    rec = recommend({'H': 0.45, 'D': 0.35, 'A': 0.20})
    assert rec['market'] == 'HD'      # doble oportunidad 1X
    assert rec['confidence'] == 'medium'


def test_recommend_double_chance_away():
    rec = recommend({'H': 0.20, 'D': 0.35, 'A': 0.45})
    assert rec['market'] == 'DA'      # doble oportunidad X2


def test_recommend_no_bet_when_unpredictable():
    rec = recommend({'H': 0.35, 'D': 0.33, 'A': 0.32})
    assert rec['market'] is None
    assert rec['confidence'] is None


def test_recommend_labels_are_strings():
    rec = recommend({'H': 0.70, 'D': 0.20, 'A': 0.10})
    assert isinstance(rec['label'], str) and len(rec['label']) > 0


def test_expected_values_positive_and_negative():
    probs = {'H': 0.50, 'D': 0.30, 'A': 0.20}
    evs = expected_values(probs, {'H': 2.50, 'D': 3.00, 'A': 4.00})
    assert evs['H'] == pytest.approx(0.25)    # 0.5*2.5-1
    assert evs['D'] == pytest.approx(-0.10)   # 0.3*3.0-1
    assert evs['A'] == pytest.approx(-0.20)   # 0.2*4.0-1


def test_expected_values_skips_invalid_odds():
    probs = {'H': 0.50, 'D': 0.30, 'A': 0.20}
    evs = expected_values(probs, {'H': 1.0, 'D': None, 'A': 0.0})
    assert evs == {}


from src.betting import recommend_combined


def test_recommend_combined_both_confident():
    rc = recommend_combined({'H': 0.65, 'D': 0.20, 'A': 0.15}, ou_prob=0.62)
    assert rc['result_rec']['market'] == 'H'
    assert rc['ou_rec']['market'] == 'Over 2.5'
    assert 'Más de 2.5' in rc['combined_label']


def test_recommend_combined_no_result_has_ou():
    rc = recommend_combined({'H': 0.35, 'D': 0.33, 'A': 0.32}, ou_prob=0.38)
    assert rc['result_rec']['market'] is None
    assert rc['ou_rec']['market'] == 'Under 2.5'
    assert rc['ou_rec']['prob'] == pytest.approx(0.62)


def test_recommend_combined_grey_zone_ou():
    rc = recommend_combined({'H': 0.65, 'D': 0.20, 'A': 0.15}, ou_prob=0.50)
    assert rc['ou_rec']['market'] is None
    assert 'incierto' in rc['ou_rec']['label'].lower()


def test_recommend_combined_ou_none_skipped():
    rc = recommend_combined({'H': 0.65, 'D': 0.20, 'A': 0.15}, ou_prob=None)
    assert rc['ou_rec'] is None
    assert rc['result_rec']['market'] == 'H'
    assert '·' not in rc['combined_label']


def test_recommend_combined_under_case():
    rc = recommend_combined({'H': 0.20, 'D': 0.35, 'A': 0.45}, ou_prob=0.35)
    assert rc['ou_rec']['market'] == 'Under 2.5'
    assert 'Menos de 2.5' in rc['combined_label']
