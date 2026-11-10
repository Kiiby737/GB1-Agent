#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('Random Search', 'M0-AL Only', 'LLM Agent OFF', 'LLM Agent ON')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--run-tag', default='')
    args = parser.parse_args()

    suffix = f'_{args.run_tag}' if args.run_tag else ''
    result_dir = args.root / '4.result'
    baseline_dir = args.root / '5.test/1.baseline/baselines'
    sources = [
        ('Random Search', baseline_dir / f'random_search_seed{args.seed}{suffix}' / 'evolution_summary.csv'),
        ('M0-AL Only', baseline_dir / f'm0_only{suffix}' / 'evolution_summary.csv'),
        ('LLM Agent OFF', result_dir / 'no_knowledge/evolution_summary.csv'),
        ('LLM Agent ON', result_dir / 'knowledge/evolution_summary.csv'),
    ]
    out = args.root / '5.test/4.result_presentation/output/fitness'
    out.mkdir(parents=True, exist_ok=True)
    batches = []
    for name, path in sources:
        batch = pd.read_csv(path)
        batch = batch.loc[batch['round'].le(args.rounds)].copy()
        batch['method'] = name
        batches.append(batch)
    table = pd.concat(batches, ignore_index=True)
    initial = pd.read_csv(args.root / '4.result/agent_initial_labeled.csv')
    initial_best = float(initial['fitness'].max())

    rows = []
    changes = []
    for method in METHODS:
        data = table.loc[table['method'].eq(method)].sort_values('round')
        assert len(data) == args.rounds, f'{method}: 轮次数量不正确'
        for item in data.itertuples(index=False):
            rows.append({
                '方法': method, '实验轮次': int(item.round),
                '新增实验数': int(item.n_measured),
                '每轮最高Fitness': float(item.round_best_fitness),
                '每轮平均Fitness': float(item.round_mean_fitness),
                '累计最高Fitness': float(item.agent_best_so_far),
                '高于WT数量': int(item.count_above_wt),
                '高于WT比例': int(item.count_above_wt) / int(item.n_measured),
            })
        first, last = data.iloc[0], data.iloc[-1]
        changes.append({
            '方法': method, '轮数': len(data),
            '首轮最高Fitness': float(first['round_best_fitness']),
            '末轮最高Fitness': float(last['round_best_fitness']),
            '最高Fitness变化_末轮减首轮': float(last['round_best_fitness'] - first['round_best_fitness']),
            '首轮平均Fitness': float(first['round_mean_fitness']),
            '末轮平均Fitness': float(last['round_mean_fitness']),
            '平均Fitness变化_末轮减首轮': float(last['round_mean_fitness'] - first['round_mean_fitness']),
            '初始20条最佳Fitness': initial_best,
            '最终累计最佳Fitness': float(last['agent_best_so_far']),
            '相对初始最佳提升': float(last['agent_best_so_far'] - initial_best),
        })

    pd.DataFrame(rows).to_csv(out / 'fitness_by_round.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(changes).to_csv(out / 'fitness_improvement_summary.csv', index=False, encoding='utf-8-sig')

    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    metrics = [('round_best_fitness', '(A) Round best Fitness'),
               ('round_mean_fitness', '(B) Round mean Fitness')]
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    for ax, (field, title) in zip(axes, metrics):
        for i, method in enumerate(METHODS):
            subset = table.loc[table['method'].eq(method)].sort_values('round')
            ax.plot(subset['round'], subset[field],
                    color=['#222222', '#777777', '#277DA1', '#D86938'][i],
                    linestyle=line_styles[i], marker=markers[i], markersize=5,
                    linewidth=1.7, label=method)
        ax.set_title(title)
        ax.set_xlabel('Evolution round')
        ax.set_ylabel('Observed Fitness')
        ax.set_xticks(range(1, args.rounds + 1))
        ax.grid(axis='y', linestyle=':', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    pdf_path = out / 'fitness_progress.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {pdf_path}')
    print(f'Saved: {out / "fitness_by_round.csv"}')
    print(f'Saved: {out / "fitness_improvement_summary.csv"}')


if __name__ == '__main__':
    main()
