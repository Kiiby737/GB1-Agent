import argparse
import json

import numpy as np
import pandas as pd
from search_strategy import initialize_search, normalize_start
from sequence_context import validate_prediction_pool

from agent_paths import (
    AGENT_INITIAL_FILE, AGENT_INITIAL_METADATA, AGENT_INITIAL_SIZE,
    AGENT_INITIAL_SEED, AGENT_RESULT_DIR, INITIAL_FILE,
    ORACLE_FILE, POOL_PREDICTIONS, TEST_FILE, MODEL_FILE,
)


def sample_initial(n_initial, seed):
    pool = pd.read_csv(POOL_PREDICTIONS, usecols=['variant'])
    pool['variant'] = pool['variant'].astype(str).str.upper().str.strip()
    assert pool['variant'].is_unique, 'Duplicate variants in prediction pool'

    m0 = set(pd.read_csv(INITIAL_FILE, usecols=['variant'])['variant'].astype(str).str.upper().str.strip())
    test = set(pd.read_csv(TEST_FILE, usecols=['variant'])['variant'].astype(str).str.upper().str.strip())
    assert not (m0 & test), 'M0 training labels overlap with the held-out test set'
    eligible = sorted(set(pool['variant']) - m0 - test)
    assert len(eligible) >= n_initial, f'Only {len(eligible)} eligible variants for {n_initial} initial samples'

    # Sampling uses only sequence IDs, never predictions or Oracle Fitness.
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(eligible), size=n_initial, replace=False)
    selected = pd.DataFrame({'variant': chosen})

    oracle = pd.read_csv(ORACLE_FILE, usecols=['AACombo', 'Fitness'])
    oracle = oracle.rename(columns={'AACombo': 'variant', 'Fitness': 'fitness'})
    oracle['variant'] = oracle['variant'].astype(str).str.upper().str.strip()
    assert oracle['variant'].is_unique, 'Duplicate variants in Oracle'
    labeled = selected.merge(oracle, on='variant', how='left', validate='one_to_one')
    assert labeled['fitness'].notna().all(), 'Some initial variants have no Oracle Fitness'
    assert labeled['variant'].is_unique
    assert not (set(labeled['variant']) & (m0 | test))
    return labeled, len(eligible)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-initial', type=int, default=AGENT_INITIAL_SIZE)
    parser.add_argument('--seed', type=int, default=AGENT_INITIAL_SEED)
    parser.add_argument('--start-variant', '--start-sequence', type=normalize_start, default=None)
    args = parser.parse_args()
    assert args.n_initial >= 1, 'Require at least one initial experiment'
    validate_prediction_pool(POOL_PREDICTIONS, INITIAL_FILE, MODEL_FILE)

    if AGENT_INITIAL_FILE.exists():
        existing = pd.read_csv(AGENT_INITIAL_FILE)
        if len(existing) != args.n_initial:
            raise ValueError(
                f'Existing initialization has {len(existing)} records, requested {args.n_initial}. '
                'Do not resample into an existing experiment branch.'
            )
        if AGENT_INITIAL_METADATA.exists():
            meta = json.loads(AGENT_INITIAL_METADATA.read_text(encoding='utf-8'))
            assert meta['seed'] == args.seed, 'Seed changed: refusing to replace an existing initialization'
        print(f'Existing independent Agent initial set preserved: {AGENT_INITIAL_FILE} ({len(existing)} variants)')
        initialize_search(args.start_variant)
        return

    labeled, eligible_count = sample_initial(args.n_initial, args.seed)
    AGENT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(AGENT_INITIAL_FILE, index=False)
    meta = {
        'method': 'uniform_without_replacement_from_eligible_predicted_pool',
        'n_initial': args.n_initial,
        'seed': args.seed,
        'eligible_pool_count': eligible_count,
        'm0_training_label_source': str(INITIAL_FILE),
        'held_out_source': str(TEST_FILE),
        'oracle_used_only_for_selected_initial_variants': True,
        'both_knowledge_modes_share_this_initial_set': True,
    }
    AGENT_INITIAL_METADATA.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Initialized {args.n_initial} independent Agent experiments from {eligible_count} eligible variants')
    print(f'Labels: {AGENT_INITIAL_FILE}')
    print(f'Metadata: {AGENT_INITIAL_METADATA}')
    print(f'Initial measured mean Fitness={labeled.fitness.mean():.6f}, best={labeled.fitness.max():.6f}')
    initialize_search(args.start_variant)


if __name__ == '__main__':
    main()
