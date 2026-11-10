"""KG-independent radius-two search domain and experimental allocation."""
import hashlib
from itertools import combinations, product

import pandas as pd

AA = 'ACDEFGHIKLMNPQRSTVWY'
POSITIONS = (39, 40, 41, 54)
BATCH_SIZE = 20
TWO_SITE_TARGET = 6
MAX_NEW_CONTROLS = 4


def distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def neighbors(anchor, order):
    for sites in combinations(range(4), order):
        choices = [[a for a in AA if a != anchor[i]] for i in sites]
        for residues in product(*choices):
            chars = list(anchor)
            for i, aa in zip(sites, residues):
                chars[i] = aa
            yield ''.join(chars)


def domain(parents, eligible):
    """Assign each ID to the nearest parent, resolving ties by parent order."""
    assigned = {}
    for order in (1, 2):
        for anchor in parents:
            for variant in neighbors(anchor, order):
                if variant in eligible and variant not in assigned:
                    assigned[variant] = (anchor, order)
    return assigned


def domain_hash(assigned):
    text = '\n'.join(f'{v},{a},{d}' for v, (a, d) in sorted(assigned.items()))
    return hashlib.sha256(text.encode()).hexdigest()


def allocation(single_count, double_count, top_k=BATCH_SIZE):
    if single_count + double_count < top_k:
        raise ValueError('The common radius-two domain cannot fill the experiment batch')
    preferred = round(TWO_SITE_TARGET * top_k / BATCH_SIZE)
    two = min(double_count, max(preferred, top_k - single_count))
    return {1: top_k - two, 2: two}


def candidate_pool(predictions, parents, observed, heldout):
    eligible = set(predictions.variant) - set(observed) - set(heldout)
    assigned = domain(parents, eligible)
    frame = predictions.loc[predictions.variant.isin(assigned)].copy()
    frame['anchor_variant'] = frame.variant.map(lambda v: assigned[v][0])
    frame['mutation_distance_to_anchor'] = frame.variant.map(lambda v: assigned[v][1])
    frame['mutation_positions'] = frame.apply(lambda r: '+'.join(
        str(POSITIONS[i]) for i in range(4) if r.variant[i] != r.anchor_variant[i]), axis=1)
    return frame.sort_values(['predicted_fitness', 'variant'], ascending=[False, True]).reset_index(drop=True)


def select_batch(frame, observed, parents, ordered=None, required=(), top_k=BATCH_SIZE):
    """Common quotas; ranking and required hypothesis tests are caller decisions."""
    if not frame.variant.is_unique:
        raise ValueError('Duplicate candidate IDs')
    rows = frame.set_index('variant').to_dict('index')
    if ordered is None:
        ordered = frame.sort_values(['predicted_fitness', 'variant'], ascending=[False, True]).variant.tolist()
    if set(ordered) != set(rows) or len(ordered) != len(rows):
        raise ValueError('Ranking must contain each candidate exactly once')
    counts = frame.mutation_distance_to_anchor.value_counts()
    targets = allocation(int(counts.get(1, 0)), int(counts.get(2, 0)), top_k)
    selected, chosen = [], set()
    used = {1: 0, 2: 0}
    measured = set(observed)
    max_controls = round(MAX_NEW_CONTROLS * top_k / BATCH_SIZE)

    def add(v):
        if v in chosen:
            return True
        row = rows[v]
        order = int(row['mutation_distance_to_anchor'])
        if used[order] >= targets[order] or len(selected) >= top_k:
            return False
        if row.get('purpose', 'test') == 'control':
            if sum(rows[x].get('purpose') == 'control' for x in selected) >= max_controls:
                return False
            ref = row.get('reference_variant', '')
            if not ref or distance(v, ref) != 1 or ref not in measured | chosen:
                return False
        selected.append(v)
        chosen.add(v)
        used[order] += 1
        return True

    for v in dict.fromkeys(required):
        if v not in rows or not add(v):
            raise ValueError(f'Mandatory test cannot fit the common allocation: {v}')
    # Every method uses the same parent coverage and mutation-order targets.
    for anchor in parents:
        options = [v for v in ordered if rows[v]['anchor_variant'] == anchor]
        for v in options:
            if add(v):
                break
    for order in (2, 1):
        for v in ordered:
            if int(rows[v]['mutation_distance_to_anchor']) == order:
                add(v)
    # A control can become valid after its reference was added in the first pass.
    for v in ordered:
        add(v)
    if len(selected) != top_k:
        raise ValueError('Candidate proposals cannot fill the shared quotas; regenerate valid test candidates')
    picked = frame.set_index('variant').loc[selected].reset_index()
    return picked, targets
