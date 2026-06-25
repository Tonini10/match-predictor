"""Bet recommendation from model probabilities.

Statistical tool only — does not guarantee outcomes.
"""

MARKET_LABELS = {
    'H': 'Local gana (1)',
    'D': 'Empate (X)',
    'A': 'Visitante gana (2)',
    'HD': 'Doble oportunidad 1X (local o empate)',
    'DA': 'Doble oportunidad X2 (empate o visitante)',
    'HA': 'Doble oportunidad 12 (local o visitante)',
    None: 'Sin apuesta — partido muy incierto',
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


def recommend_combined(result_probs, ou_prob):
    """Combine result and over/under recommendations into a single response."""
    result_rec = recommend(result_probs)

    if ou_prob is None:
        ou_rec = None
    elif ou_prob >= 0.60:
        ou_rec = {'market': 'Over 2.5', 'label': 'Más de 2.5 goles', 'prob': ou_prob}
    elif ou_prob <= 0.40:
        ou_rec = {'market': 'Under 2.5', 'label': 'Menos de 2.5 goles', 'prob': round(1 - ou_prob, 3)}
    else:
        ou_rec = {'market': None, 'label': 'Goles inciertos', 'prob': None}

    result_part = result_rec['label'] if result_rec['market'] else 'Sin recomendación'

    if ou_rec is None:
        combined_label = result_part
    else:
        combined_label = f"{result_part}  ·  {ou_rec['label']}"

    return {
        'result_rec': result_rec,
        'ou_rec': ou_rec,
        'combined_label': combined_label,
    }
