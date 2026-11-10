import argparse
from pathlib import Path

import pandas as pd

CASES = {2: 1, 3: 3}
DEFAULT_ROOT = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4.9')


def select_variants(df, n):
    """自动提取预测绝对误差最大的 n 条有效单点突变案例。"""
    candidates = df.copy()
    candidates['variant'] = candidates['variant'].astype(str).str.strip().str.upper()
    candidates['anchor_variant'] = candidates['anchor_variant'].astype(str).str.strip().str.upper()
    valid_pair = candidates.apply(
        lambda row: len(row['anchor_variant']) == 4
        and len(row['variant']) == 4
        and sum(a != b for a, b in zip(row['anchor_variant'], row['variant'])) == 1,
        axis=1,
    )
    candidates = candidates.loc[valid_pair].copy()
    candidates['_abs_error'] = (
        pd.to_numeric(candidates['predicted_fitness'], errors='coerce')
        - pd.to_numeric(candidates['fitness'], errors='coerce')
    ).abs()
    candidates = candidates.dropna(subset=['_abs_error'])
    candidates = candidates.sort_values(
        ['_abs_error', 'variant'], ascending=[False, True], kind='mergesort'
    )
    if len(candidates) < n:
        raise ValueError(f'Only {len(candidates)} valid single-site cases, need {n}')
    return tuple(candidates.head(n)['variant'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--mode', choices=('knowledge', 'no_knowledge'), default='knowledge')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (args.root / '5.test' / '3.failure_analysis' / 'prediction_error_cases.csv')
    records = []

    for round_num, n_cases in CASES.items():
        path = args.root / '4.result' / args.mode / f'round_{round_num:03d}' / 'experiment_results.csv'
        df = pd.read_csv(path)
        variants = select_variants(df, n_cases)
        for variant in variants:
            hit = df.loc[df['variant'].astype(str).str.upper().eq(variant)]
            if len(hit) != 1:
                raise ValueError(f'{path}: {variant} should appear exactly once, found {len(hit)}')
            row = hit.iloc[0]
            predicted = float(row['predicted_fitness'])
            actual = float(row['fitness'])
            error = predicted - actual
            records.append({
                '实验轮次': f'Round {round_num}',
                '变体': variant,
                '预测 Fitness': round(predicted, 6),
                '真实 Fitness': round(actual, 6),
                '预测误差': round(error, 6),
                '绝对误差': round(abs(error), 6),
                '预测偏差类型': '高估' if error > 0 else ('低估' if error < 0 else '无偏差'),
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, encoding='utf-8-sig')
    print(f'Saved: {output}')
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
