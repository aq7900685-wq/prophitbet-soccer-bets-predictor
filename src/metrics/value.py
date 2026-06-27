import numpy as np
from typing import Dict, List, Optional


# Default minimum expected value (per unit stake) required to actually place a bet.
# An EV of 0 is break-even; the margin protects against vig/variance and model error.
DEFAULT_MIN_EDGE = 0.05

# Decision labels emitted by the evaluator.
BET = 'BET'
SHOP = 'SHOP'
SKIP = 'SKIP'


def evaluate_value_bet(
        probs: np.ndarray,
        odds: np.ndarray,
        labels: List[str],
        min_edge: float = DEFAULT_MIN_EDGE
) -> Dict:
    """ Evaluates a single match and returns a value-betting decision.

    The decision is made on expected value (EV), not on raw win probability. For each mutually
    exclusive outcome, EV = prob*odds - 1 (expected profit per unit stake) and the probability
    edge = prob - implied_prob, where implied_prob = 1/odds. The outcome with the highest EV is
    the "value pick". The ruling is:

        * BET  if best EV >= min_edge          (clears the break-even gate with a safety margin)
        * SHOP if 0 <= best EV < min_edge       (positive but thin; a better price could promote it)
        * SKIP if best EV < 0                    (negative expectation at the offered price)

    A consistency flag marks whether the value pick is also the most-likely outcome (argmax prob).
    When it is not, the value pick is a longshot relative to the model's central estimate, which the
    caller should surface so the favorite-trap (high win% but no price value) is avoided.

    :param probs: 1D array of outcome probabilities (must sum to ~1).
    :param odds: 1D array of decimal odds aligned with `probs`.
    :param labels: Outcome labels aligned with `probs` (e.g. ['1', 'X', '2']).
    :param min_edge: Minimum EV required to rule BET.
    :returns: A decision dict (see keys below).
    """

    probs = np.asarray(probs, dtype=np.float64)
    odds = np.asarray(odds, dtype=np.float64)

    if probs.shape != odds.shape or probs.shape[0] != len(labels):
        raise ValueError('probs, odds and labels must share the same length.')

    implied = 1.0 / odds
    ev = probs * odds - 1.0
    edge = probs - implied

    best = int(np.argmax(ev))
    likely = int(np.argmax(probs))
    best_ev = float(ev[best])

    if best_ev >= min_edge:
        decision = BET
    elif best_ev >= 0.0:
        decision = SHOP
    else:
        decision = SKIP

    return {
        'pick': labels[best],
        'pick_index': best,
        'ev': best_ev,
        'edge': float(edge[best]),
        'prob': float(probs[best]),
        'implied': float(implied[best]),
        'break_even_odds': float(1.0 / probs[best]) if probs[best] > 0.0 else float('inf'),
        'decision': decision,
        'consistent': best == likely,
        'likely': labels[likely],
        'ev_per_outcome': {labels[i]: float(ev[i]) for i in range(len(labels))}
    }


def evaluate_value_bets(
        y_prob: np.ndarray,
        odds: np.ndarray,
        labels: List[str],
        min_edge: float = DEFAULT_MIN_EDGE
) -> List[Dict]:
    """ Vectorized wrapper over `evaluate_value_bet` for a batch of matches.

    :param y_prob: 2D array of shape (n_matches, n_outcomes).
    :param odds: 2D array of shape (n_matches, n_outcomes), aligned with `y_prob`.
    :param labels: Outcome labels aligned with the columns of `y_prob`.
    :param min_edge: Minimum EV required to rule BET.
    :returns: A list of decision dicts, one per match.
    """

    y_prob = np.asarray(y_prob, dtype=np.float64)
    odds = np.asarray(odds, dtype=np.float64)

    if y_prob.shape != odds.shape:
        raise ValueError(f'y_prob {y_prob.shape} and odds {odds.shape} must share the same shape.')

    return [evaluate_value_bet(y_prob[i], odds[i], labels=labels, min_edge=min_edge) for i in range(y_prob.shape[0])]


def format_decision(decision: Dict) -> str:
    """ Renders a decision dict as a compact table-cell string, e.g. "BET 2 (EV +7%)".

    An asterisk is appended when the value pick is not the most-likely outcome, signalling that the
    bet is a value/longshot play rather than backing the favorite.
    """

    pick = decision['pick']
    ev_pct = decision['ev'] * 100.0
    flag = '' if decision['consistent'] else '*'
    return f'{decision["decision"]} {pick}{flag} (EV {ev_pct:+.0f}%)'


def decision_columns(decisions: List[Dict]) -> Dict[str, List]:
    """ Builds parallel column lists ('Value', 'EV%', 'Edge%', 'Decision') from decision dicts,
        suitable for assignment into a results DataFrame or table.
    """

    value, ev_pct, edge_pct, ruling = [], [], [], []
    for d in decisions:
        flag = '' if d['consistent'] else '*'
        value.append(f'{d["pick"]}{flag}')
        ev_pct.append(round(d['ev'] * 100.0, 1))
        edge_pct.append(round(d['edge'] * 100.0, 1))
        ruling.append(d['decision'])
    return {'Value': value, 'EV%': ev_pct, 'Edge%': edge_pct, 'Decision': ruling}
