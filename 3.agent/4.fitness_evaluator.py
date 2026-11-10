import pandas as pd

from agent_paths import get_round_arg
from search_strategy import validate_candidates


def main():
    p = get_round_arg()
    candidates = pd.read_csv(p.designed)
    validate_candidates(p, candidates)
    scores = pd.read_csv(p.predictions,
                         usecols=['variant', 'predicted_fitness', 'uncertainty'])
    result = candidates.merge(scores, on='variant', how='left', validate='one_to_one')
    if result[['predicted_fitness', 'uncertainty']].isna().any().any():
        raise ValueError('Candidate missing from frozen prediction pool')
    result = result.sort_values('predicted_fitness', ascending=False).reset_index(drop=True)
    result.insert(0, 'prediction_rank', range(1, len(result) + 1))
    result['model_version'] = p.model_name
    result.to_csv(p.scored, index=False)
    print(f'Round {p.round_num} | {p.model_name} | {len(result)} candidates -> {p.scored}')
    print(result[['prediction_rank', 'variant', 'predicted_fitness', 'uncertainty']]
          .head(10).to_string(index=False))


if __name__ == '__main__':
    main()
