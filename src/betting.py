"""Bet recommendation from model probabilities.

Statistical tool only — does not guarantee outcomes.
"""


def compute_all_markets(result_probs, over_1_5=None, over_2_5=None, over_3_5=None, btts=None):
    """Return all betting markets derived from model probabilities.

    result_probs: dict with keys 'H', 'D', 'A'
    Returns a dict with every market probability and its label.
    """
    ph = result_probs.get('H', 0.0)
    pd_ = result_probs.get('D', 0.0)
    pa = result_probs.get('A', 0.0)

    # Normalise so probabilities sum to 1 (guard against float drift)
    total = ph + pd_ + pa
    if total > 0:
        ph, pd_, pa = ph / total, pd_ / total, pa / total

    dnb_total = ph + pa if (ph + pa) > 0 else 1.0

    markets = {
        # ── Resultado 1X2 ────────────────────────────────────────────
        '1':  {'label': 'Local gana (1)',       'prob': ph,  'group': 'resultado'},
        'X':  {'label': 'Empate (X)',            'prob': pd_, 'group': 'resultado'},
        '2':  {'label': 'Visitante gana (2)',    'prob': pa,  'group': 'resultado'},
        # ── Doble oportunidad ───────────────────────────────────────
        '1X': {'label': 'Doble oportunidad 1X', 'prob': ph + pd_, 'group': 'doble'},
        'X2': {'label': 'Doble oportunidad X2', 'prob': pd_ + pa, 'group': 'doble'},
        '12': {'label': 'Doble oportunidad 12', 'prob': ph + pa,  'group': 'doble'},
        # ── Draw No Bet ──────────────────────────────────────────────
        'DNB1': {'label': 'DNB Local',     'prob': ph / dnb_total, 'group': 'dnb'},
        'DNB2': {'label': 'DNB Visitante', 'prob': pa / dnb_total, 'group': 'dnb'},
        # ── Goles Over/Under ─────────────────────────────────────────
        'O15': {'label': 'Más de 1.5 goles',   'prob': over_1_5,              'group': 'goles'},
        'U15': {'label': 'Menos de 1.5 goles', 'prob': 1 - over_1_5 if over_1_5 is not None else None, 'group': 'goles'},
        'O25': {'label': 'Más de 2.5 goles',   'prob': over_2_5,              'group': 'goles'},
        'U25': {'label': 'Menos de 2.5 goles', 'prob': 1 - over_2_5 if over_2_5 is not None else None, 'group': 'goles'},
        'O35': {'label': 'Más de 3.5 goles',   'prob': over_3_5,              'group': 'goles'},
        'U35': {'label': 'Menos de 3.5 goles', 'prob': 1 - over_3_5 if over_3_5 is not None else None, 'group': 'goles'},
        # ── Ambos marcan ─────────────────────────────────────────────
        'BTTS_Y': {'label': 'Ambos marcan: Sí', 'prob': btts,                                   'group': 'btts'},
        'BTTS_N': {'label': 'Ambos marcan: No', 'prob': 1 - btts if btts is not None else None, 'group': 'btts'},
        # ── Hándicap asiático ─────────────────────────────────────────
        'AH_H05': {'label': 'Local -0.5',     'prob': ph,        'group': 'handicap'},
        'AH_H05P': {'label': 'Local +0.5',    'prob': ph + pd_,  'group': 'handicap'},
        'AH_A05': {'label': 'Visitante -0.5', 'prob': pa,        'group': 'handicap'},
        'AH_A05P': {'label': 'Visitante +0.5','prob': pd_ + pa,  'group': 'handicap'},
    }
    return markets


def best_bet(markets, threshold=0.60):
    """Return the single market with highest probability above threshold, or None."""
    candidates = [
        (k, v) for k, v in markets.items()
        if v['prob'] is not None and v['prob'] >= threshold
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda kv: kv[1]['prob'])


def expected_value(prob, odd):
    """EV = model_prob × decimal_odd − 1."""
    if prob is None or odd is None or odd <= 1.0:
        return None
    return round(prob * odd - 1.0, 3)


# ── Backward-compatible helpers ──────────────────────────────────────────────

def recommend(probs):
    _ORDER = 'HDA'
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, top_p = ordered[0]
    labels = {'H': 'Local gana (1)', 'D': 'Empate (X)', 'A': 'Visitante gana (2)',
              'HD': 'Doble oportunidad 1X', 'DA': 'Doble oportunidad X2',
              'HA': 'Doble oportunidad 12', None: 'Sin recomendación'}
    if top_p >= 0.60:
        return {'market': top, 'label': labels[top], 'confidence': 'high'}
    if top_p >= 0.50:
        return {'market': top, 'label': labels[top], 'confidence': 'medium'}
    if top_p >= 0.40:
        second = ordered[1][0]
        market = ''.join(sorted(top + second, key=_ORDER.index))
        return {'market': market, 'label': labels[market], 'confidence': 'medium'}
    return {'market': None, 'label': labels[None], 'confidence': None}


def recommend_combined(result_probs, ou_prob):
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
    combined_label = result_part if ou_rec is None else f"{result_part}  ·  {ou_rec['label']}"
    return {'result_rec': result_rec, 'ou_rec': ou_rec, 'combined_label': combined_label}
