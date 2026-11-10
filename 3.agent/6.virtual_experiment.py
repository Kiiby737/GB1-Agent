import pandas as pd

from agent_paths import ORACLE_FILE, INITIAL_FILE, get_round_arg, read_observed
from search_strategy import validate_candidates
from sequence_context import training_template, sequence_for_variant


def main():
    p = get_round_arg()
    if p.experiments.exists():
        raise FileExistsError(f'Completed experiments already exist: {p.experiments}')
    selected = pd.read_csv(p.topk)
    if len(selected) != 20 or not selected['variant'].is_unique:
        raise ValueError('Expected 20 unique new experiments')
    validate_candidates(p, selected)
    measured = set(read_observed(p)['variant'])
    held_out = set(pd.read_csv(p.test, usecols=['variant'])['variant'])
    assert not set(selected['variant']) & (measured | held_out), 'Already tested / held-out variant in Top-k'

    oracle = pd.read_csv(ORACLE_FILE, usecols=['AACombo', 'Fitness'])
    oracle = oracle.rename(columns={'AACombo': 'variant', 'Fitness': 'fitness'})
    result = selected.merge(oracle, on='variant', how='left', validate='one_to_one')
    template = training_template(INITIAL_FILE)
    result['mutant_seq'] = result['variant'].map(lambda v: sequence_for_variant(v, template))
    assert result['fitness'].notna().all(), 'Selected variant missing from GB1 Oracle'
    result['prediction_error'] = result['predicted_fitness'] - result['fitness']
    result.to_csv(p.experiments, index=False)

    print(f'Round {p.round_num}: measured {len(result)} variants')
    print(f'True Fitness mean={result.fitness.mean():.4f}, best={result.fitness.max():.4f}')
    print(f'Saved: {p.experiments}')


if __name__ == '__main__':
    main()
