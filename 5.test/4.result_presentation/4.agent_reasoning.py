#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WT = 'VDGV'
POSITIONS = (39, 40, 41, 54)


def mutations(reference, variant):
    return ', '.join(f'{a}{p}{b}' for a, b, p in zip(reference, variant, POSITIONS) if a != b) or '-'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--mode', choices=['knowledge', 'no_knowledge'], default='knowledge')
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()

    out = args.root / '5.test/4.result_presentation/output' / args.mode
    out.mkdir(parents=True, exist_ok=True)
    site_rows, hypothesis_rows, reason_rows, combination_rows = [], [], [], []
    markdown = [f'# GB1 Agent 推理记录（{args.mode}）',
                '以下“实验前依据”来自原始 Agent 输出；真实 Fitness 仅用于实验后的结果核对。']

    for r in range(1, args.rounds + 1):
        folder = args.root / '4.result' / args.mode / f'round_{r:03d}'
        analysis = json.loads((folder / 'data_analysis.json').read_text(encoding='utf-8'))
        hypothesis_data = json.loads((folder / 'hypotheses.json').read_text(encoding='utf-8'))
        review = json.loads((folder / 'scientific_review.json').read_text(encoding='utf-8'))
        topk = pd.read_csv(folder / 'topk_recommendations.csv').sort_values('prediction_rank')
        experiment = pd.read_csv(folder / 'experiment_results.csv')

        history = pd.DataFrame(analysis['observed']['site_stats'])
        enriched = pd.DataFrame(analysis['predicted']['top100_vs_pool'])
        merged = enriched.merge(history, on='mutation', how='left')
        for row in merged.to_dict('records'):
            site_rows.append({'轮次': r, **row})

        assessments = {a['hypothesis_id']: a for a in review['hypothesis_assessments']}
        hypotheses = {h['id']: h for h in hypothesis_data['hypotheses']}
        markdown.extend([f'\n## Round {r}',
                         f'**Agent 总结：** {hypothesis_data.get("summary", "")}'])

        focus = merged.sort_values('top_vs_pool_ratio', ascending=False).head(5)
        markdown.append('\n**模型 Top-100 中相对候选池富集的位点/替换（仅说明模型偏好，不证明因果有益）：**')
        for record in focus.to_dict('records'):
            markdown.append(f'- {record["mutation"]}：富集比 {record["top_vs_pool_ratio"]:.2f}；'
                            f'历史样本数 {int(record["count"]):d}')

        for h in hypothesis_data['hypotheses']:
            kg = h.get('knowledge_design') or {}
            assessment = assessments.get(h['id'], {})
            row = {
                '轮次': r, '假设编号': h['id'], '搜索策略': h.get('strategy', ''),
                '亲本': h.get('anchor_variant', ''), '假设内容': h.get('hypothesis', ''),
                '已测实验依据': h.get('observed_evidence', ''),
                '预测模型依据': h.get('predicted_evidence', ''),
                '设计指导': h.get('design_guidance', ''),
                '建议配对': json.dumps(h.get('matched_comparison', {}), ensure_ascii=False),
                '证伪设计': h.get('falsification_test', ''),
                'KG解释': kg.get('llm_knowledge_rationale', h.get('knowledge_rationale', '')),
                'Critic假设判断': assessment.get('assessment', ''),
                'Critic判断依据': assessment.get('reason', ''),
            }
            hypothesis_rows.append(row)
            markdown.extend([f'\n**{h["id"]}：{h.get("hypothesis", "")}**',
                             f'- 真实实验依据：{row["已测实验依据"]}',
                             f'- 模型预测依据：{row["预测模型依据"]}',
                             f'- 设计指导：{row["设计指导"]}',
                             f'- KG 依据：{row["KG解释"]}' if row['KG解释'] else '',
                             f'- Critic：{row["Critic假设判断"]}；{row["Critic判断依据"]}'])

        for row in topk.to_dict('records'):
            variant = row['variant']
            anchor = row['anchor_variant']
            item = {
                '轮次': r, '排名': int(row['prediction_rank']), '变体': variant,
                '假设编号': row['hypothesis_id'], '亲本': anchor,
                '相对亲本替换': mutations(anchor, variant),
                '相对WT突变组合': mutations(WT, variant),
                '预测Fitness': row['predicted_fitness'],
                '设计理由': row.get('reason', ''),
                'Critic科学解释': row.get('scientific_reason', ''),
                'Critic局限性': row.get('limitations', ''),
            }
            reason_rows.append(item)

        markdown.append('\n**Top-3 推荐理由（实验前记录）：**')
        for item in reason_rows[-len(topk):][:3]:
            markdown.append(f'- {item["变体"]}：{item["相对亲本替换"]}；'
                            f'{item["Critic科学解释"]}')

        experiment = experiment.sort_values('fitness', ascending=False)
        for row in experiment.to_dict('records'):
            v = row['variant']
            anchor = row['anchor_variant']
            if sum(a != b for a, b in zip(WT, v)) < 2:
                continue
            combination_rows.append({
                '轮次': r, '变体': v, '亲本': anchor,
                '相对亲本新增替换': mutations(anchor, v),
                '相对WT的多位点突变': mutations(WT, v),
                '假设编号': row['hypothesis_id'],
                '假设中的突变设计依据': hypotheses[row['hypothesis_id']].get('design_guidance', ''),
                '实验前设计原因': row.get('reason', ''),
                '实验前Critic解释': row.get('scientific_reason', ''),
                '预测Fitness': row['predicted_fitness'],
                '真实Fitness_实验后': row['fitness'],
            })

        best = experiment.iloc[0]
        markdown.append(f'\n**实验后最佳：** {best["variant"]}，真实 Fitness = {best["fitness"]:.4f}。')

    outputs = {
        'site_screening_statistics.csv': site_rows,
        'hypotheses_trace.csv': hypothesis_rows,
        'topk_reasoning.csv': reason_rows,
        'mutation_combination_trace.csv': combination_rows,
    }
    for filename, records in outputs.items():
        pd.DataFrame(records).to_csv(out / filename, index=False, encoding='utf-8-sig')
        print(f'Saved: {out / filename}')

    markdown_path = out / 'agent_reasoning_report.md'
    markdown_path.write_text('\n'.join(x for x in markdown if x) + '\n', encoding='utf-8')
    print(f'Saved: {markdown_path}')


if __name__ == '__main__':
    main()
