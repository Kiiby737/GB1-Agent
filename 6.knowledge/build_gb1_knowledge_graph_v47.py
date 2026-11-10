#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4.7')
INPUT_CSV = ROOT / '2.train/hybrid_30_40_30_r9/M0_AL_labeled.csv'
OUT = ROOT / '6.knowledge'

WT = 'VDGV'
POSITIONS = (39, 40, 41, 54)

AA = {
    'A': ('nonpolar', 'neutral', 'small', 1.8),
    'C': ('polar', 'neutral', 'small', 2.5),
    'D': ('polar', 'negative', 'medium', -3.5),
    'E': ('polar', 'negative', 'medium', -3.5),
    'F': ('nonpolar', 'neutral', 'large', 2.8),
    'G': ('nonpolar', 'neutral', 'small', -0.4),
    'H': ('polar', 'pH_dependent', 'medium', -3.2),
    'I': ('nonpolar', 'neutral', 'medium', 4.5),
    'K': ('polar', 'positive', 'large', -3.9),
    'L': ('nonpolar', 'neutral', 'medium', 3.8),
    'M': ('nonpolar', 'neutral', 'medium', 1.9),
    'N': ('polar', 'neutral', 'medium', -3.5),
    'P': ('nonpolar', 'neutral', 'small', -1.6),
    'Q': ('polar', 'neutral', 'medium', -3.5),
    'R': ('polar', 'positive', 'large', -4.5),
    'S': ('polar', 'neutral', 'small', -0.8),
    'T': ('polar', 'neutral', 'small', -0.7),
    'V': ('nonpolar', 'neutral', 'small', 4.2),
    'W': ('nonpolar', 'neutral', 'large', -0.9),
    'Y': ('polar', 'neutral', 'large', -1.3),
}

CONSERVATIVE_GROUPS = {
    'aliphatic': 'AVILM', 'aromatic': 'FWY', 'acidic': 'DE',
    'basic': 'KR', 'amide': 'NQ', 'hydroxyl': 'ST',
}
RULES = {
    'R1': 'Only 20 canonical amino acids are allowed at each GB1 site.',
    'R2': 'Prefer one-site changes relative to an already measured parent.',
    'R3': 'Use conservative versus radical substitutions as contrasting hypotheses.',
    'R4': 'A beneficial substitution is supported only in its measured sequence background.',
    'R5': 'Combining beneficial substitutions requires new measurements because of epistasis.',
    'R6': 'Count mutations relative to WT separately from changes relative to a parent.',
    'R7': 'Protein-level 4-AA variants cannot contain a stop codon; DNA design needs codon checks.',
}

nodes = {}
edges = {}  # 按完整边属性去重；同一端点但位点属性不同的边保留


def node(node_id, node_type, **attrs):
    nodes[node_id] = {'id': node_id, 'type': node_type, **attrs}


def edge(source, relation, target, **attrs):
    key = (source, relation, target, tuple(sorted(attrs.items())))
    edges[key] = {'source': source, 'relation': relation, 'target': target, **attrs}


def is_conservative(a, b):
    return any(a in group and b in group for group in CONSERVATIVE_GROUPS.values())


def sequence_context(varying, background):
    """用固定残基定义背景；varying 是允许变化的四位点下标。"""
    fixed = [i for i in range(4) if i not in varying]
    cid = f"context:{'-'.join(str(POSITIONS[i]) for i in varying)}:{background}"
    node(cid, 'SequenceContext', varying_positions=[POSITIONS[i] for i in varying],
         fixed_positions=[POSITIONS[i] for i in fixed], background=background)
    for i in varying:
        edge(cid, 'varies_at', f'position:{POSITIONS[i]}')
    for i, aa in zip(fixed, background):
        edge(cid, 'fixes_position', f'position:{POSITIONS[i]}', amino_acid=aa)
        edge(cid, 'fixes_amino_acid', f'aa:{aa}', position=POSITIONS[i])
    return cid


def main():
    df = pd.read_csv(INPUT_CSV, usecols=['variant', 'fitness'])
    df['variant'] = df['variant'].str.upper().str.strip()
    df = df.drop_duplicates('variant').reset_index(drop=True)
    measured = dict(zip(df['variant'], df['fitness'].astype(float)))

    for aa, (polarity, charge, size, hydropathy) in AA.items():
        node(f'aa:{aa}', 'AminoAcid', symbol=aa, hydropathy=hydropathy,
             polarity=polarity, charge=charge, size=size)
        for attr, value in [('polarity', polarity), ('charge', charge), ('size', size)]:
            prop = f'property:{attr}:{value}'
            node(prop, 'Property', property=attr, value=value)
            edge(f'aa:{aa}', 'has_property', prop)

    for position, wt_aa in zip(POSITIONS, WT):
        node(f'position:{position}', 'Position', position=position, wt_amino_acid=wt_aa)
        edge(f'position:{position}', 'wildtype_amino_acid', f'aa:{wt_aa}')

    for group, members in CONSERVATIVE_GROUPS.items():
        for a, b in combinations(members, 2):
            edge(f'aa:{a}', 'conservative_substitution', f'aa:{b}',
                 group=group, symmetric=True)

    for rid, description in RULES.items():
        node(f'rule:{rid}', 'MutationRule', rule=rid, description=description)

    combination_rows = []
    for variant, fitness in measured.items():
        mutations = [(POSITIONS[i], f'{WT[i]}{POSITIONS[i]}{aa}')
                     for i, aa in enumerate(variant) if WT[i] != aa]
        node(f'variant:{variant}', 'Variant', sequence=variant,
             mutations_from_wt=len(mutations), measured=True)
        node(f'observation:{variant}', 'FitnessObservation', fitness=float(fitness),
             evidence='known_experiment')
        edge(f'variant:{variant}', 'has_observation', f'observation:{variant}')

        for position, mutation in mutations:
            wt_aa, alt = mutation[0], mutation[-1]
            mid = f'mutation:{mutation}'
            node(mid, 'Mutation', name=mutation, position=position,
                 from_aa=wt_aa, to_aa=alt,
                 conservative=is_conservative(wt_aa, alt))
            edge(f'variant:{variant}', 'contains_mutation', mid)
            edge(mid, 'occurs_at', f'position:{position}')
            edge(mid, 'changes_from', f'aa:{wt_aa}')
            edge(mid, 'changes_to', f'aa:{alt}')

        # 2～4 位点的 WT 相对突变组合；组合可在不同实测变体中复用。
        for order in range(2, len(mutations) + 1):
            for subset in combinations(mutations, order):
                names = [mutation for _, mutation in subset]
                cid = 'combination:' + '+'.join(names)
                node(cid, 'MutationCombination', mutations=names, mutation_count=order)
                edge(f'variant:{variant}', 'contains_combination', cid)
                for name in names:
                    edge(cid, 'contains_mutation', f'mutation:{name}')
                combination_rows.append({
                    'variant': variant, 'combination_id': cid,
                    'mutations': '+'.join(names), 'mutation_count': order,
                    'variant_fitness': float(fitness),
                    'other_mutations': '+'.join(m for _, m in mutations if m not in names),
                })

    # 单点背景：固定另外三位；仅将已测序列组成比较对。
    contexts = defaultdict(list)
    for variant in measured:
        for i in range(4):
            contexts[(i, variant[:i] + variant[i + 1:])].append(variant)

    pair_rows = []
    directed_rows = []
    for (i, background), variants in contexts.items():
        position = POSITIONS[i]
        context_id = sequence_context((i,), background)
        for variant in variants:
            edge(f'variant:{variant}', 'has_sequence_context', context_id)

        for a, b in combinations(sorted(variants), 2):
            delta = float(measured[b] - measured[a])
            pid = f'comparison:{a}:{b}'
            substitution = f'{a[i]}{position}{b[i]}'
            node(pid, 'MeasuredSingleSiteComparison', reference_variant=a,
                 test_variant=b, position=position, background=background,
                 substitution=substitution, delta_fitness=delta,
                 reference_fitness=float(measured[a]), test_fitness=float(measured[b]),
                 evidence='observed_fitness_difference_not_global_mutation_effect')
            edge(f'variant:{a}', 'comparison_reference', pid)
            edge(pid, 'comparison_test', f'variant:{b}')
            edge(pid, 'has_sequence_context', context_id)
            edge(pid, 'occurs_at', f'position:{position}')
            pair_rows.append({
                'reference_variant': a, 'test_variant': b,
                'position': position, 'background': background,
                'substitution': substitution,
                'reference_fitness': float(measured[a]),
                'test_fitness': float(measured[b]),
                'delta_fitness': delta, 'conservative': is_conservative(a[i], b[i]),
            })

            # 保留原表的无向配对计数；另建双向查询表和双向图边。
            for ref, test in ((a, b), (b, a)):
                directed_delta = float(measured[test] - measured[ref])
                directed_substitution = f'{ref[i]}{position}{test[i]}'
                edge(f'variant:{ref}', 'single_site_transition', f'variant:{test}',
                     position=position, substitution=directed_substitution,
                     delta_fitness=directed_delta, comparison_id=pid,
                     context_id=context_id)
                directed_rows.append({
                    'reference_variant': ref, 'test_variant': test,
                    'position': position, 'background': background,
                    'substitution': directed_substitution,
                    'reference_fitness': float(measured[ref]),
                    'test_fitness': float(measured[test]),
                    'delta_fitness': directed_delta,
                    'conservative': is_conservative(ref[i], test[i]),
                    'comparison_id': pid,
                })

    # 完整四角实测的 2×2 突变方格：epsilon = F11 - F10 - F01 + F00。
    # 若 AA 对包含该位置 WT，优先使用 WT 作基线；否则按字母序选基线。
    epistasis_rows = []
    for i, j in combinations(range(4), 2):
        fixed = [k for k in range(4) if k not in (i, j)]
        groups = defaultdict(dict)
        for variant in measured:
            background = ''.join(variant[k] for k in fixed)
            groups[background][(variant[i], variant[j])] = variant

        for background, grid in groups.items():
            aas_i = sorted({a for a, _ in grid})
            aas_j = sorted({b for _, b in grid})
            for pair_i in combinations(aas_i, 2):
                for pair_j in combinations(aas_j, 2):
                    a0, a1 = (WT[i], next(x for x in pair_i if x != WT[i])) \
                        if WT[i] in pair_i else pair_i
                    b0, b1 = (WT[j], next(x for x in pair_j if x != WT[j])) \
                        if WT[j] in pair_j else pair_j
                    corners = ((a0, b0), (a1, b0), (a0, b1), (a1, b1))
                    if not all(corner in grid for corner in corners):
                        continue
                    v00, v10, v01, v11 = [grid[corner] for corner in corners]
                    f00, f10, f01, f11 = [float(measured[v]) for v in
                                           (v00, v10, v01, v11)]
                    epsilon = f11 - f10 - f01 + f00
                    sub_a = f'{a0}{POSITIONS[i]}{a1}'
                    sub_b = f'{b0}{POSITIONS[j]}{b1}'
                    eid = f'epistasis:{POSITIONS[i]}-{POSITIONS[j]}:{background}:{a0}{a1}:{b0}{b1}'
                    context_id = sequence_context((i, j), background)
                    node(eid, 'EpistasisInteraction', position_a=POSITIONS[i],
                         position_b=POSITIONS[j], substitution_a=sub_a,
                         substitution_b=sub_b, background=background,
                         baseline_variant=v00, single_a_variant=v10,
                         single_b_variant=v01, double_variant=v11,
                         f00=f00, f10=f10, f01=f01, f11=f11,
                         epsilon=epsilon, scale='raw_fitness',
                         evidence='four_measured_variants')
                    edge(eid, 'has_sequence_context', context_id)
                    edge(eid, 'interaction_position', f'position:{POSITIONS[i]}')
                    edge(eid, 'interaction_position', f'position:{POSITIONS[j]}')
                    for role, variant in [('baseline', v00), ('single_a', v10),
                                          ('single_b', v01), ('double', v11)]:
                        edge(eid, role, f'variant:{variant}')
                        edge(f'variant:{variant}', 'has_sequence_context', context_id)
                    epistasis_rows.append({
                        'position_a': POSITIONS[i], 'position_b': POSITIONS[j],
                        'background': background, 'context_id': context_id,
                        'substitution_a': sub_a, 'substitution_b': sub_b,
                        'baseline_variant': v00, 'single_a_variant': v10,
                        'single_b_variant': v01, 'double_variant': v11,
                        'f00': f00, 'f10': f10, 'f01': f01, 'f11': f11,
                        'epsilon': epsilon,
                    })

    OUT.mkdir(parents=True, exist_ok=True)
    edge_values = list(edges.values())
    pd.DataFrame(nodes.values()).to_csv(OUT / 'kg_nodes.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(edge_values).to_csv(OUT / 'kg_edges.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(pair_rows).to_csv(OUT / 'kg_measured_pairs.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(directed_rows).to_csv(OUT / 'kg_directed_pairs.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(combination_rows).to_csv(OUT / 'kg_mutation_combinations.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(epistasis_rows, columns=[
        'position_a', 'position_b', 'background', 'context_id',
        'substitution_a', 'substitution_b',
        'baseline_variant', 'single_a_variant', 'single_b_variant', 'double_variant',
        'f00', 'f10', 'f01', 'f11', 'epsilon',
    ]).to_csv(OUT / 'kg_epistasis.csv', index=False, encoding='utf-8-sig')

    graph = {'wildtype': WT, 'positions': list(POSITIONS),
             'source': str(INPUT_CSV), 'rules': RULES,
             'nodes': list(nodes.values()), 'edges': edge_values}
    (OUT / 'kg_graph.json').write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {
        'known_variants': len(measured), 'node_count': len(nodes),
        'edge_count': len(edge_values),
        'matched_single_site_pairs': len(pair_rows),
        'directed_single_site_pairs': len(directed_rows),
        'mutation_combination_memberships': len(combination_rows),
        'epistasis_interactions': len(epistasis_rows),
        'node_types': dict(Counter(n['type'] for n in nodes.values())),
        'output_directory': str(OUT),
    }
    (OUT / 'kg_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
