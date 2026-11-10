import argparse
from pathlib import Path

import pandas as pd

WT = 'VDGV'
WT_SEQ = 'MTYKLIINGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
POSITIONS = (39, 40, 41, 54)
ROOT = Path(__file__).resolve().parents[2]


def changes(reference, variant):
    return ', '.join(f'{a}{pos}{b}' for a, b, pos in zip(reference, variant, POSITIONS) if a != b) or '-'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--mode', choices=['knowledge', 'no_knowledge'], default='knowledge')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--per-round', type=int, default=2, help='每轮展示真实 Fitness 排名前 N 的变体')
    args = parser.parse_args()

    out = args.root / '5.test/4.result_presentation/output' / args.mode
    out.mkdir(parents=True, exist_ok=True)

    rows = [{
        '轮次': 'WT', '轮内排名': 0, '变体': WT,
        '相对WT突变': '-', '相对亲本突变': '-', '亲本': '-',
        '真实Fitness': None, '完整蛋白序列': WT_SEQ,
    }]
    alignments = [f'WT     {WT_SEQ}']

    for r in range(1, args.rounds + 1):
        path = args.root / '4.result' / args.mode / f'round_{r:03d}' / 'experiment_results.csv'
        data = pd.read_csv(path).sort_values('fitness', ascending=False).head(args.per_round)
        for rank, (_, item) in enumerate(data.iterrows(), 1):
            v = item['variant']
            anchor = item['anchor_variant']
            full = item['mutant_seq']
            assert ''.join(full[p - 1] for p in POSITIONS) == v, f'{v}: 全序列与四位点不匹配'
            rows.append({
                '轮次': f'Round {r}', '轮内排名': rank, '变体': v,
                '相对WT突变': changes(WT, v),
                '相对亲本突变': changes(anchor, v),
                '亲本': anchor, '真实Fitness': round(float(item['fitness']), 4),
                '完整蛋白序列': full,
            })
            mask = ''.join('^' if a != b else ' ' for a, b in zip(WT_SEQ, full))
            alignments.extend([f'R{r}-{rank:<3} {full}', f'       {mask}'])

    csv_path = out / 'sequence_examples.csv'
    txt_path = out / 'sequence_alignment.txt'
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    txt_path.write_text('\n'.join(alignments) + '\n', encoding='utf-8')
    print(f'Saved: {csv_path}\nSaved: {txt_path}')


if __name__ == '__main__':
    main()
