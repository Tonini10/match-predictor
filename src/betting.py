"""Bet recommendation from model probabilities.

Statistical tool only — does not guarantee outcomes.
"""

MARKET_LABELS = {
    'H': 'Home win (1)',
    'D': 'Draw (X)',
    'A': 'Away win (2)',
    'HD': 'Double chance 1X (home win or draw)',
    'DA': 'Double chance X2 (draw or away win)',
    'HA': 'Double chance 12 (home or away win)',
    None: 'No bet — match too unpredictable',
}

_ORDER = 'HDA'


def recommend(probs):
    """Threshold-based recommendation from {H, D, A} probabilities.

    >= 0.60 -> single outcome, high confidence
    >= 0.50 -> single outcome, medium confidence
    >= 0.40 -> double chance with the two most likely outcomes, medium
    <  0.40 -> no bet
    """
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, top_p = ordered[0]
    if top_p >= 0.60:
        return {'market': top, 'label': MARKET_LABELS[top], 'confidence': 'high'}
    if top_p >= 0.50:
        return {'market': top, 'label': MARKET_LABELS[top], 'confidence': 'medium'}
    if top_p >= 0.40:
        second = ordered[1][0]
        market = ''.join(sorted(top + second, key=_ORDER.index))
        return {'market': market, 'label': MARKET_LABELS[market], 'confidence': 'medium'}
    return {'market': None, 'label': MARKET_LABELS[None], 'confidence': None}


def expected_values(probs, odds):
    """EV per 1X2 market for user-supplied decimal odds.

    EV = model probability x decimal odd - 1. Markets with missing or
    invalid odds (<= 1.0) are omitted.
    """
    out = {}
    for market in ('H', 'D', 'A'):
        odd = odds.get(market)
        if odd is None or odd <= 1.0:
            continue
        out[market] = round(probs.get(market, 0.0) * odd - 1.0, 3)
    return out
