#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd

CASES = {2: 1, 3: 3}
DEFAULT_ROOT = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4.9')
POSITIONS = (39, 40, 41, 54)


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


def substitution(parent, variant):
    if len(parent) != 4 or len(variant) != 4:
        raise ValueError(f'Invalid 4-site variants: {parent}, {variant}')
    diff = [i for i, (a, b) in enumerate(zip(parent, variant)) if a != b]
    if len(diff) != 1:
        raise ValueError(f'Not a single-site pair: {parent} -> {variant}')
    i = diff[0]
    return f'{parent[i]}{POSITIONS[i]}{variant[i]}'


def get_known_fitness(root, current_round):
    """Only initial and strictly earlier-round observations; no later-round leakage."""
    files = [root / '4.result' / 'agent_initial_labeled.csv']
    files += [root / '4.result' / 'knowledge' / f'round_{r:03d}' / 'experiment_results.csv'
              for r in range(1, current_round)]
    known = {}
    for file in files:
        df = pd.read_csv(file, usecols=['variant', 'fitness'])
        for item in df.itertuples(index=False):
            variant = str(item.variant).strip().upper()
            fitness = float(item.fitness)
            if variant in known and abs(known[variant] - fitness) > 1e-9:
                raise ValueError(f'Conflicting labels for {variant}: {file}')
            known[variant] = fitness
    return known


def direction(x):
    if pd.isna(x):
        return None
    return 1 if x > 0 else -1 if x < 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (args.root / '5.test' / '3.failure_analysis' / 'kg_background_cases.csv')
    records = []

    for round_num, n_cases in CASES.items():
        path = args.root / '4.result' / 'knowledge' / f'round_{round_num:03d}' / 'experiment_results.csv'
        df = pd.read_csv(path)
        variants = select_variants(df, n_cases)
        known = get_known_fitness(args.root, round_num)
        for variant in variants:
            hit = df.loc[df['variant'].astype(str).str.upper().eq(variant)]
            if len(hit) != 1:
                raise ValueError(f'{path}: {variant} should appear exactly once, found {len(hit)}')
            row = hit.iloc[0]
            parent = str(row['anchor_variant']).strip().upper()
            mutation = substitution(parent, variant)
            if parent not in known:
                raise ValueError(f'Parent {parent} has no pre-round measured fitness for Round {round_num}')
            reported = row.get('kg_substitution', None)
            if pd.notna(reported) and str(reported) != mutation:
                raise ValueError(f'KG substitution mismatch for {variant}: {reported} != {mutation}')
            parent_fitness = known[parent]
            actual = float(row['fitness'])
            delta = actual - parent_fitness
            kg_median = row['kg_other_background_median_delta']
            kg_median = float(kg_median) if pd.notna(kg_median) else float('nan')
            kg_n = row['kg_other_background_n']
            kg_n = int(kg_n) if pd.notna(kg_n) else 0
            kg_dir, actual_dir = direction(kg_median), direction(delta)
            records.append({
                '实验轮次': f'Round {round_num}',
                '变体': variant,
                '亲本变体': parent,
                '单点突变': mutation,
                '历史背景配对数': kg_n,
                '历史突变效应中位数': round(kg_median, 6),
                '亲本真实 Fitness': round(parent_fitness, 6),
                '变体真实 Fitness': round(actual, 6),
                '当前背景真实 Fitness 差值': round(delta, 6),
                '历史与当前效应方向一致': ('是' if kg_dir == actual_dir else '否') if kg_dir is not None else '无历史证据',
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, encoding='utf-8-sig')
    print(f'Saved: {output}')
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
