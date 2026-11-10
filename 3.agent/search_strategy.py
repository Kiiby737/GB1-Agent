"""Common radius-two multi-parent search; KG changes evidence, never permissions."""
import hashlib
import json
import math
from pathlib import Path
from itertools import combinations

import pandas as pd

import agent_paths as ap
from sequence_context import validate_prediction_pool
from common_search import domain, domain_hash, allocation, neighbors

AA = set('ACDEFGHIKLMNPQRSTVWY')
WT = 'VDGV'
PROTOCOL = 'seeded-common-radius2-1'
BATCH_SIZE = 20
MAX_NEW_CONTROLS = 4
MAX_PARENTS = 3


def normalize_start(value=None):
    value = str(value or WT).strip().upper() or WT
    if len(value) != 4 or not set(value) <= AA:
        raise ValueError('Starting variant must have four canonical amino acids')
    return value


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def start_file():
    return ap.AGENT_RESULT_DIR / 'search_start.json'


def input_fingerprints():
    validate_prediction_pool(ap.POOL_PREDICTIONS, ap.INITIAL_FILE, ap.MODEL_FILE)
    files = {'m0': ap.INITIAL_FILE, 'initial20': ap.AGENT_INITIAL_FILE,
             'heldout': ap.TEST_FILE, 'prediction_pool': ap.POOL_PREDICTIONS,
             'oracle': ap.ORACLE_FILE}
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in files.items()}


def load_start():
    start = json.loads(start_file().read_text(encoding='utf-8'))
    if start['protocol'] != PROTOCOL:
        raise ValueError('Old search protocol. Use a NEW GB1_AGENT_RESULT_DIR (preserve old runs).')
    if start['input_sha256'] != input_fingerprints():
        raise ValueError('Experiment input files changed')
    return start


def initialize_search(requested=None):
    if start_file().exists():
        start = load_start()
        if requested is not None and normalize_start(requested) != start['variant']:
            raise ValueError('Cannot change the starting sequence in an existing run')
        return start
    if any(ap.AGENT_RESULT_DIR.glob('*/round_*/experiment_results.csv')):
        raise ValueError('Old rounds exist without this protocol. Use a fresh result directory')
    seed = normalize_start(requested)
    heldout = set(pd.read_csv(ap.TEST_FILE, usecols=['variant'])['variant'])
    if seed in heldout:
        raise ValueError('Seed is in the independent test split')
    history = pd.concat([pd.read_csv(ap.INITIAL_FILE, usecols=['variant', 'fitness']),
                         pd.read_csv(ap.AGENT_INITIAL_FILE, usecols=['variant', 'fitness'])])
    matches = history.loc[history['variant'] == seed, 'fitness']
    if len(matches) > 1:
        raise ValueError('Conflicting seed labels')
    if len(matches):
        fitness, source, queries = float(matches.iloc[0]), 'already_visible_history', 0
    else:
        selected = []
        for chunk in pd.read_csv(ap.ORACLE_FILE, usecols=['AACombo', 'Fitness'], chunksize=50000):
            selected.extend(chunk.loc[chunk['AACombo'] == seed, 'Fitness'].tolist())
        if len(selected) != 1:
            raise ValueError(f'Seed {seed} not uniquely found in Oracle')
        fitness, source, queries = float(selected[0]), 'single_seed_oracle_query', 1
    if not math.isfinite(fitness):
        raise ValueError('Seed label is not finite')
    start = {'protocol': PROTOCOL, 'variant': seed, 'fitness': fitness, 'source': source,
             'seed_oracle_queries': queries, 'same_seed_for_both_modes': True,
             'count_as_new_round_experiment': False,
             'input_sha256': input_fingerprints(),
             'parent_policy': 'best_fitness_then_radius2_coverage_with_single_comparison_max3'}
    ap.AGENT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start_file().write_text(json.dumps(start, indent=2), encoding='utf-8')
    return start


def single_neighbors(anchor):
    return {anchor[:i] + aa + anchor[i + 1:] for i in range(4)
            for aa in AA if aa != anchor[i]}


def search_state(paths, minimum_candidates=BATCH_SIZE, *, branch_dir=None, observed=None):
    start = load_start()
    slots = ap.kg_validation_slots() if paths.knowledge_mode == 'on' else 0
    observed = ap.read_observed(paths) if observed is None else observed
    heldout = set(pd.read_csv(paths.test, usecols=['variant'])['variant'])
    pool = set(pd.read_csv(paths.predictions, usecols=['variant'])['variant'])
    eligible = pool - set(observed['variant']) - heldout
    trajectory = [{'variant': start['variant'], 'fitness': float(start['fitness']),
                   'discovered_round': 0}]
    seen = {start['variant']}
    history_fingerprints = {}
    for number in range(1, paths.round_num):
        folder = (Path(branch_dir) / f'round_{number:03d}' if branch_dir else
                  ap.existing_round_folder(number))
        previous = json.loads((folder / 'search_state.json').read_text(encoding='utf-8'))
        if previous['protocol'] != PROTOCOL or previous['start_variant'] != start['variant']:
            raise ValueError('Previous round belongs to a different protocol or seed')
        if previous['kg_validation_slots'] != slots:
            raise ValueError('Do not change the KG reserved-slot policy within a branch')
        parents = set(previous['parent_variants'])
        if not parents <= seen:
            raise ValueError('Previous parents do not belong to seed-descendant lineage')
        result_path = folder / 'experiment_results.csv'
        data = pd.read_csv(result_path)
        if len(data) != BATCH_SIZE or not data['variant'].is_unique:
            raise ValueError('Each round requires 20 unique measured candidates')
        for row in data.to_dict('records'):
            variant, anchor = str(row['variant']), str(row['anchor_variant'])
            method = str(row.get('design_method', ''))  # Baseline may not store this column
            distance = hamming(variant, anchor)
            if (anchor not in parents or variant in seen or not math.isfinite(float(row['fitness']))
                    or distance not in (1, 2)):
                raise ValueError(f'Invalid seed-descendant experiment lineage: {anchor} -> {variant}')
            seen.add(variant)
            trajectory.append({'variant': variant, 'fitness': float(row['fitness']),
                               'discovered_round': number})
        history_fingerprints[str(number)] = hashlib.sha256(result_path.read_bytes()).hexdigest()

    ranked = sorted(trajectory, key=lambda r: (-r['fitness'], r['discovered_round'], r['variant']))
    candidates = []
    for row in ranked:
        singles = single_neighbors(row['variant']) & eligible
        available = (singles | set(neighbors(row['variant'], 2))) & eligible
        if singles:
            candidates.append({**row, 'eligible_neighbors': len(available),
                               'eligible_single_site_neighbors': len(singles), '_neighbor_set': available})
    if not candidates:
        raise ValueError('No measured seed descendant has a single-site comparison available')

    # Keep the highest-fitness parent, then maximize *new* candidate coverage.
    # This prevents three overlapping neighborhoods from excluding a fourth useful parent.
    chosen = [candidates[0]]
    covered = set(chosen[0]['_neighbor_set'])
    while len(chosen) < min(MAX_PARENTS, len(candidates)):
        remaining = [x for x in candidates if x['variant'] not in
                     {c['variant'] for c in chosen}]
        next_parent = max(remaining, key=lambda x: (
            len(x['_neighbor_set'] - covered),
            min(hamming(x['variant'], c['variant']) for c in chosen) >= 2,
            x['fitness'], -x['discovered_round'], x['variant']))
        chosen.append(next_parent)
        covered |= next_parent['_neighbor_set']

    # If the best primary cannot form a sufficient group, also check other measured
    # descendants before reporting exhaustion. There are at most 20*round+1 descendants.
    if len(covered) < minimum_candidates:
        best_coverage = len(covered)
        for size in range(1, min(MAX_PARENTS, len(candidates)) + 1):
            for group in combinations(candidates, size):
                combined = set().union(*(x['_neighbor_set'] for x in group))
                best_coverage = max(best_coverage, len(combined))
                if len(combined) >= minimum_candidates:
                    chosen = list(group)
                    covered = combined
                    break
            if len(covered) >= minimum_candidates:
                break

    if len(covered) < minimum_candidates:
        raise ValueError(f'No set of up to {MAX_PARENTS} measured parents has '
                         f'{minimum_candidates} eligible radius-two candidates; maximum found {best_coverage}')
    chosen = sorted(chosen, key=lambda x: (-x['fitness'], x['discovered_round'], x['variant']))
    neighbor_union = covered
    chosen = [{key: value for key, value in row.items() if key != '_neighbor_set'}
              for row in chosen]
    primary = chosen[0]
    assigned = domain([x['variant'] for x in chosen], eligible)
    singles = sum(order == 1 for _, order in assigned.values())
    doubles = len(assigned) - singles
    targets = allocation(singles, doubles, minimum_candidates)
    state = {
        'protocol': PROTOCOL, 'round': paths.round_num, 'knowledge_mode': paths.knowledge_mode,
        'kg_validation_slots': slots, 'start_variant': start['variant'],
        'parent_variant': primary['variant'], 'parent_fitness': primary['fitness'],
        'parent_discovered_round': primary['discovered_round'],
        'parent_variants': [x['variant'] for x in chosen], 'parents': chosen,
        'trajectory_observed_count': len(trajectory), 'eligible_neighbor_count': len(neighbor_union),
        'minimum_candidates': minimum_candidates, 'previous_round_sha256': history_fingerprints,
        'eligible_single_site_count': singles, 'eligible_two_site_count': doubles,
        'two_site_target': targets[2], 'admissible_pool_sha256': domain_hash(assigned),
        'selection_policy': 'common-radius2-target6-no-kg-exclusive-slots',
    }
    path = paths.folder / 'search_state.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != state:
            raise ValueError('Search state changed while resuming this round')
    else:
        paths.folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def validate_candidates(paths, candidates, state=None):
    state = state or search_state(paths)
    parents = set(state['parent_variants'])
    forbidden = (set(ap.read_observed(paths)['variant']) |
                 set(pd.read_csv(paths.test, usecols=['variant'])['variant']))
    rows = candidates.to_dict('records')
    predictions = pd.read_csv(paths.predictions, usecols=['variant'])
    assigned = domain(state['parent_variants'], set(predictions.variant) - forbidden)
    for row in rows:
        variant = str(row['variant']).strip().upper()
        anchor = str(row['anchor_variant']).strip().upper()
        method = str(row['design_method'])
        if len(variant) != 4 or not set(variant) <= AA or anchor not in parents:
            raise ValueError(f'Invalid candidate/seed-descendant parent: {variant}, {anchor}')
        distance = hamming(variant, anchor)
        if distance not in (1, 2) or assigned.get(variant) != (anchor, distance):
            raise ValueError(f'{variant}: candidate leaves the common canonical radius-two domain')
        if state['knowledge_mode'] == 'off' and str(row.get('kg_evidence_id', '') or '').strip() not in ('', 'nan'):
            raise ValueError('OFF candidates cannot carry retrieved KG evidence')
        evidence_id = str(row.get('kg_evidence_id', '') or '').strip()
        if state['knowledge_mode'] == 'on' and evidence_id not in ('', 'nan'):
            records = pd.read_csv(paths.folder / 'kg_combination_proposals.csv', keep_default_na=False)
            accepted = {(r.anchor_variant, r.variant, r.evidence_id) for r in records.itertuples(index=False)}
            if distance != 2 or (anchor, variant, evidence_id) not in accepted:
                raise ValueError('KG annotation has no matching historical proposal evidence')
        if variant in forbidden:
            raise ValueError(f'{variant}: already measured or held out')
    return state
