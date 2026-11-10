import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POSITIONS = (39, 40, 41, 54)


def single_mutation(parent, variant):
    if not isinstance(parent, str) or len(parent) != 4:
        return ''
    return ', '.join(f'{a}{p}{b}' for a, b, p in zip(parent, variant, POSITIONS) if a != b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--mode', choices=['knowledge', 'no_knowledge'], default='knowledge')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args()

    out = args.root / '5.test/4.result_presentation/output' / args.mode
    out.mkdir(parents=True, exist_ok=True)
    batches = []

    for r in range(1, args.rounds + 1):
        folder = args.root / '4.result' / args.mode / f'round_{r:03d}'
        topk = pd.read_csv(folder / 'topk_recommendations.csv')
        measured = pd.read_csv(folder / 'experiment_results.csv', usecols=['variant', 'fitness'])
        topk = topk.sort_values('prediction_rank').head(args.top_k)
        topk = topk.merge(measured, on='variant', how='left', validate='one_to_one')

        # ON 有 KG 证据列；OFF 没有。只导出实际存在的信息。
        fields = {
            'prediction_rank': '推荐排名', 'variant': '候选序列',
            'hypothesis_id': '假设编号', 'purpose': '实验用途',
            'strategy': '搜索策略', 'anchor_variant': '亲本序列',
            'reference_variant': '对照序列', 'changed_position': '突变位置',
            'predicted_fitness': '预测Fitness', 'uncertainty': '预测不确定性',
            'fitness': '真实Fitness', 'design_method': '设计来源',
            'reason': '设计理由', 'scientific_reason': 'Scientific Critic解释',
            'limitations': '证据局限', 'kg_substitution': 'KG突变类型',
            'kg_other_background_n': 'KG其他背景数量',
            'kg_other_background_median_delta': 'KG历史效应中位数',
        }
        selected = topk[[c for c in fields if c in topk.columns]].rename(columns=fields).copy()
        selected.insert(0, '轮次', r)
        selected.insert(
            selected.columns.get_loc('亲本序列') + 1,
            '相对亲本替换',
            [single_mutation(a, v) for a, v in zip(topk['anchor_variant'], topk['variant'])],
        )
        selected.to_csv(out / f'topk_round_{r:03d}.csv', index=False, encoding='utf-8-sig')
        batches.append(selected)

    result = pd.concat(batches, ignore_index=True)
    path = out / 'topk_all_rounds.csv'
    result.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'Saved: {path} ({len(result)} recommendations)')


if __name__ == '__main__':
    main()
