#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import median

import pandas as pd

from agent_paths import KG_DIR, read_observed
from knowledge_validation import sequence_differences, epistasis_records

WT = 'VDGV'
POSITIONS = (39, 40, 41, 54)
AA = 'ACDEFGHIKLMNPQRSTVWY'
CONSERVATIVE_GROUPS = {
    'aliphatic': 'AVILM', 'aromatic': 'FWY', 'acidic': 'DE',
    'basic': 'KR', 'amide': 'NQ', 'hydroxyl': 'ST',
}


def single_site_change(reference, candidate):
    if len(reference) != 4 or len(candidate) != 4:
        return None
    changed = [i for i in range(4) if reference[i] != candidate[i]]
    return changed[0] if len(changed) == 1 else None


def conservative(a, b):
    return any(a in g and b in g for g in CONSERVATIVE_GROUPS.values())


def measured_pairs(observed):
    """每组真实同背景单点配对只保留一次，维持旧 CSV 和计数。"""
    fitness = dict(zip(observed['variant'], observed['fitness']))
    grouped = defaultdict(list)
    for v in fitness:
        for i in range(4):
            grouped[i, v[:i] + v[i + 1:]].append(v)
    rows = []
    for (i, background), group in grouped.items():
        for a, b in combinations(sorted(group), 2):
            rows.append({
                'reference_variant': a, 'test_variant': b,
                'position': POSITIONS[i], 'background': background,
                'substitution': f'{a[i]}{POSITIONS[i]}{b[i]}',
                'reference_fitness': float(fitness[a]),
                'test_fitness': float(fitness[b]),
                'delta_fitness': float(fitness[b] - fitness[a]),
                'conservative': conservative(a[i], b[i]),
            })
    return pd.DataFrame(rows, columns=[
        'reference_variant', 'test_variant', 'position', 'background',
        'substitution', 'reference_fitness', 'test_fitness',
        'delta_fitness', 'conservative',
    ])


def directed_pairs(pairs):
    """双向索引：反向 ΔFitness 取相反数；不将一对实验当作两份独立证据。"""
    rows = []
    for pair in pairs.itertuples(index=False):
        a, b = pair.reference_variant, pair.test_variant
        i = POSITIONS.index(int(pair.position))
        for ref, test, ref_f, test_f in (
            (a, b, pair.reference_fitness, pair.test_fitness),
            (b, a, pair.test_fitness, pair.reference_fitness),
        ):
            rows.append({
                'reference_variant': ref, 'test_variant': test,
                'position': int(pair.position), 'background': pair.background,
                'substitution': f'{ref[i]}{pair.position}{test[i]}',
                'reference_fitness': float(ref_f), 'test_fitness': float(test_f),
                'delta_fitness': float(test_f - ref_f),
                'conservative': conservative(ref[i], test[i]),
                'comparison_id': f'comparison:{a}:{b}',
            })
    return pd.DataFrame(rows, columns=[
        'reference_variant', 'test_variant', 'position', 'background',
        'substitution', 'reference_fitness', 'test_fitness',
        'delta_fitness', 'conservative', 'comparison_id',
    ])


def mutation_combinations(observed):
    """组合是相对 WT 的突变集合，Fitness 仍属于完整变体而非子组合。"""
    rows = []
    for variant, fitness in observed[['variant', 'fitness']].itertuples(index=False, name=None):
        names = [f'{WT[i]}{POSITIONS[i]}{aa}'
                 for i, aa in enumerate(variant) if WT[i] != aa]
        for order in range(2, len(names) + 1):
            for subset in combinations(names, order):
                rows.append({
                    'variant': variant, 'combination_id': 'combination:' + '+'.join(subset),
                    'mutations': '+'.join(subset), 'mutation_count': order,
                    'variant_fitness': float(fitness),
                    'other_mutations': '+'.join(x for x in names if x not in subset),
                })
    return pd.DataFrame(rows, columns=[
        'variant', 'combination_id', 'mutations', 'mutation_count',
        'variant_fitness', 'other_mutations',
    ])


def epistasis_interactions(observed):
    """只计算四角全部已测的二位点上位性；与 ver4.7 新 KG0 的方向约定一致。"""
    return pd.DataFrame(epistasis_records(observed), columns=[
        'position_a', 'position_b', 'background', 'context_id',
        'substitution_a', 'substitution_b',
        'baseline_variant', 'single_a_variant', 'single_b_variant', 'double_variant',
        'f00', 'f10', 'f01', 'f11', 'epsilon',
    ])


def save_graph_snapshot(base, observed, pairs, directed, combo, epistasis, folder, round_num):
    """KG0 + 当前分支截至目标轮之前的真实标签；同步写入背景、双向边和上位性。"""
    nodes = {n['id']: n for n in base['nodes']}
    edges = {(
        e['source'], e['relation'], e['target'],
        tuple(sorted((k, v) for k, v in e.items() if k not in ('source', 'relation', 'target')))
    ): e for e in base['edges']}

    def add_node(nid, node_type, **attrs):
        if nid not in nodes:
            nodes[nid] = {'id': nid, 'type': node_type, **attrs}

    def add_edge(source, relation, target, **attrs):
        key = (source, relation, target, tuple(sorted(attrs.items())))
        edges[key] = {'source': source, 'relation': relation, 'target': target, **attrs}

    def add_context(varying, background):
        fixed = [k for k in range(4) if k not in varying]
        cid = f"context:{'-'.join(str(POSITIONS[k]) for k in varying)}:{background}"
        add_node(cid, 'SequenceContext', varying_positions=[POSITIONS[k] for k in varying],
                 fixed_positions=[POSITIONS[k] for k in fixed], background=background)
        for k in varying:
            add_edge(cid, 'varies_at', f'position:{POSITIONS[k]}')
        for k, aa in zip(fixed, background):
            add_edge(cid, 'fixes_position', f'position:{POSITIONS[k]}', amino_acid=aa)
            add_edge(cid, 'fixes_amino_acid', f'aa:{aa}', position=POSITIONS[k])
        return cid

    for variant, fitness in observed[['variant', 'fitness']].itertuples(index=False, name=None):
        vid, oid = f'variant:{variant}', f'observation:{variant}'
        mutated = [(i, WT[i], aa) for i, aa in enumerate(variant) if WT[i] != aa]
        add_node(vid, 'Variant', sequence=variant, mutations_from_wt=len(mutated), measured=True)
        add_node(oid, 'FitnessObservation', fitness=float(fitness),
                 evidence='known_experiment_prior_to_target_round')
        add_edge(vid, 'has_observation', oid)
        for i, old, new in mutated:
            pos = POSITIONS[i]
            name = f'{old}{pos}{new}'
            mid = f'mutation:{name}'
            add_node(mid, 'Mutation', name=name, position=pos, from_aa=old,
                     to_aa=new, conservative=conservative(old, new))
            add_edge(vid, 'contains_mutation', mid)
            add_edge(mid, 'occurs_at', f'position:{pos}')
            add_edge(mid, 'changes_from', f'aa:{old}')
            add_edge(mid, 'changes_to', f'aa:{new}')

        for i in range(4):
            cid = add_context((i,), variant[:i] + variant[i + 1:])
            add_edge(vid, 'has_sequence_context', cid)

    for row in combo.itertuples(index=False):
        add_node(row.combination_id, 'MutationCombination',
                 mutations=row.mutations.split('+'), mutation_count=int(row.mutation_count))
        add_edge(f'variant:{row.variant}', 'contains_combination', row.combination_id)
        for name in row.mutations.split('+'):
            add_edge(row.combination_id, 'contains_mutation', f'mutation:{name}')

    for pair in pairs.itertuples(index=False):
        a, b = pair.reference_variant, pair.test_variant
        cid = f'comparison:{a}:{b}'
        i = POSITIONS.index(int(pair.position))
        context_id = add_context((i,), pair.background)
        add_node(cid, 'MeasuredSingleSiteComparison', reference_variant=a,
                 test_variant=b, position=int(pair.position), background=pair.background,
                 substitution=pair.substitution, delta_fitness=float(pair.delta_fitness),
                 reference_fitness=float(pair.reference_fitness),
                 test_fitness=float(pair.test_fitness),
                 evidence='observed_fitness_difference_not_global_mutation_effect')
        add_edge(f'variant:{a}', 'comparison_reference', cid)
        add_edge(cid, 'comparison_test', f'variant:{b}')
        add_edge(cid, 'has_sequence_context', context_id)
        add_edge(cid, 'occurs_at', f'position:{pair.position}')

    for row in directed.itertuples(index=False):
        context_id = f'context:{row.position}:{row.background}'
        add_edge(f'variant:{row.reference_variant}', 'single_site_transition',
                 f'variant:{row.test_variant}', position=int(row.position),
                 substitution=row.substitution, delta_fitness=float(row.delta_fitness),
                 comparison_id=row.comparison_id, context_id=context_id)

    for row in epistasis.itertuples(index=False):
        pa, pb, background = int(row.position_a), int(row.position_b), row.background
        i, j = POSITIONS.index(pa), POSITIONS.index(pb)
        context_id = add_context((i, j), background)
        sa, sb = row.substitution_a, row.substitution_b
        eid = f'epistasis:{pa}-{pb}:{background}:{sa[0]}{sa[-1]}:{sb[0]}{sb[-1]}'
        add_node(eid, 'EpistasisInteraction', position_a=pa, position_b=pb,
                 substitution_a=sa, substitution_b=sb, background=background,
                 baseline_variant=row.baseline_variant,
                 single_a_variant=row.single_a_variant,
                 single_b_variant=row.single_b_variant,
                 double_variant=row.double_variant,
                 f00=float(row.f00), f10=float(row.f10),
                 f01=float(row.f01), f11=float(row.f11),
                 epsilon=float(row.epsilon), scale='raw_fitness',
                 evidence='four_measured_variants')
        add_edge(eid, 'has_sequence_context', context_id)
        add_edge(eid, 'interaction_position', f'position:{pa}')
        add_edge(eid, 'interaction_position', f'position:{pb}')
        for role, variant in (
            ('baseline', row.baseline_variant),
            ('single_a', row.single_a_variant),
            ('single_b', row.single_b_variant),
            ('double', row.double_variant),
        ):
            add_edge(eid, role, f'variant:{variant}')
            add_edge(f'variant:{variant}', 'has_sequence_context', context_id)

    snapshot = {**{k: v for k, v in base.items() if k not in ('nodes', 'edges')},
                'snapshot_for_round': round_num, 'observed_count': len(observed),
                'nodes': list(nodes.values()), 'edges': list(edges.values())}
    path = folder / 'kg_graph_snapshot.json'
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')
    return path


def build_knowledge(paths):
    """KG0 为固定 M0 数据；动态图谱只加入本分支目标轮之前的已测数据。"""
    graph_path = KG_DIR / 'kg_graph.json'
    graph = json.loads(graph_path.read_text(encoding='utf-8'))
    pair_source = KG_DIR / 'kg_measured_pairs.csv'
    base_pair_count = len(pd.read_csv(pair_source, keep_default_na=False))

    known = read_observed(paths)
    pairs = measured_pairs(known)
    directed = directed_pairs(pairs)
    combo = mutation_combinations(known)
    epistasis = epistasis_interactions(known)

    paths.folder.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(paths.folder / 'kg_observed_pairs.csv', index=False)
    directed.to_csv(paths.folder / 'kg_observed_directed_pairs.csv', index=False)
    combo.to_csv(paths.folder / 'kg_observed_combinations.csv', index=False)
    epistasis.to_csv(paths.folder / 'kg_observed_epistasis.csv', index=False)
    load_pairs.cache_clear()
    snapshot_path = save_graph_snapshot(
        graph, known, pairs, directed, combo, epistasis, paths.folder, paths.round_num)

    aa = {
        n['symbol']: {'polarity': n['polarity'], 'charge': n['charge'],
                      'size': n['size'], 'hydropathy': float(n['hydropathy'])}
        for n in graph['nodes'] if n['type'] == 'AminoAcid'
    }
    ordered = pairs.sort_values('delta_fitness')

    def pair_examples(df):
        return df[['reference_variant', 'test_variant', 'position',
                   'substitution', 'delta_fitness']].round(4).to_dict('records')

    strongest_epi = (epistasis.assign(abs_epsilon=epistasis['epsilon'].abs())
                     .sort_values('abs_epsilon', ascending=False).head(10))
    epi_examples = strongest_epi[[
        'position_a', 'position_b', 'background', 'substitution_a',
        'substitution_b', 'baseline_variant', 'single_a_variant',
        'single_b_variant', 'double_variant', 'epsilon',
    ]].round(4).to_dict('records')

    return {
        'type': 'GB1_own_measured_fitness_knowledge_graph',
        'source_graph': str(graph_path),
        'graph_snapshot': str(snapshot_path),
        'm0_history_variants': len(pd.read_csv(paths.m0_history, usecols=['variant'])),
        'source_pairs': str(pair_source),
        'target_round': paths.round_num,
        'base_graph_pairs': base_pair_count,
        'observed_variant_count': len(known),
        'observed_one_site_pairs': len(pairs),
        'observed_directed_pairs': len(directed),
        'observed_mutation_combinations': combo['combination_id'].nunique(),
        'observed_epistasis_interactions': len(epistasis),
        'rules': graph['rules'],
        'amino_acids': aa,
        'conservative_groups_heuristic': CONSERVATIVE_GROUPS,
        'largest_measured_positive_pairs': pair_examples(ordered.tail(10).iloc[::-1]),
        'largest_measured_negative_pairs': pair_examples(ordered.head(10)),
        'largest_absolute_epistasis': epi_examples,
        'evidence_policy': ('Only M0 and earlier measured labels from the current branch may enter KG. '
                            'Reverse transitions are one experiment viewed in two directions. '
                            'Observed single-site effects and four-corner epistasis are background-dependent. '
                            'Predicted fitness is never a verified KG observation.'),
    }


@lru_cache(maxsize=8)
def load_pairs(path):
    return pd.read_csv(path, keep_default_na=False).to_dict('records')


def evaluate_substitution(knowledge, paths, reference, candidate, max_examples=2):
    """沿用旧返回字段，增加双向同位点证据、序列背景与上位性证据。"""
    reference = str(reference).upper().strip()
    candidate = str(candidate).upper().strip()
    idx = single_site_change(reference, candidate)
    if idx is None:
        return None

    old, new = reference[idx], candidate[idx]
    position = POSITIONS[idx]
    background = reference[:idx] + reference[idx + 1:]
    context_id = f'context:{position}:{background}'
    substitution = f'{old}{position}{new}'
    p = knowledge['amino_acids']

    history = []
    for pair in load_pairs(str(paths.folder / 'kg_observed_directed_pairs.csv')):
        if int(pair['position']) != position or pair['substitution'] != substitution:
            continue
        if pair['background'] == background:
            continue  # 候选尚未测量；只用其他固定三位背景的对照数据
        history.append({
            'evidence_id': f"measured:{pair['reference_variant']}:{pair['test_variant']}",
            'variant_a': pair['reference_variant'], 'variant_b': pair['test_variant'],
            'reference_variant': pair['reference_variant'],
            'test_variant': pair['test_variant'],
            'substitution': substitution,
            'reference_fitness': float(pair['reference_fitness']),
            'test_fitness': float(pair['test_fitness']),
            'other_background': pair['background'],
            'comparison_anchor': reference,
            'background_differences_to_anchor': sequence_differences(
                pair['reference_variant'], reference, position),
            'delta_in_proposed_direction': round(float(pair['delta_fitness']), 5),
        })

    epistasis_history = []
    for item in load_pairs(str(paths.folder / 'kg_observed_epistasis.csv')):
        pa, pb = int(item['position_a']), int(item['position_b'])
        if position not in (pa, pb):
            continue
        focal_sub = item['substitution_a'] if position == pa else item['substitution_b']
        if focal_sub[0] == old and focal_sub[-1] == new:
            direction = 1
        elif focal_sub[0] == new and focal_sub[-1] == old:
            direction = -1
        else:
            continue
        other_position = pb if position == pa else pa
        other_sub = item['substitution_b'] if position == pa else item['substitution_a']
        fixed = [k for k in range(4) if POSITIONS[k] not in (pa, pb)]
        same_fixed_background = item['background'] == ''.join(reference[k] for k in fixed)
        epistasis_history.append({
            'evidence_id': (f"epistasis:{pa}-{pb}:{item['background']}:"
                            f"{item['substitution_a'][0]}{item['substitution_a'][-1]}:"
                            f"{item['substitution_b'][0]}{item['substitution_b'][-1]}"),
            'other_position': other_position,
            'other_substitution': other_sub,
            'two_site_background': item['background'],
            'same_fixed_background': same_fixed_background,
            'baseline_variant': item['baseline_variant'],
            'single_a_variant': item['single_a_variant'],
            'single_b_variant': item['single_b_variant'],
            'double_variant': item['double_variant'],
            'epsilon_in_proposed_direction': round(direction * float(item['epsilon']), 5),
            'canonical_epsilon': float(item['epsilon']),
            'focal_substitution': substitution,
            'focal_direction': direction,
            'epsilon_scale': 'raw_fitness',
        })

    deltas = [r['delta_in_proposed_direction'] for r in history]
    epistasis_history.sort(key=lambda x: (
        -int(x['same_fixed_background']),
        -abs(x['epsilon_in_proposed_direction']),
        x['evidence_id'],
    ))
    return {
        'reference_variant': reference, 'test_variant': candidate,
        'changed_position': position, 'substitution': substitution,
        'sequence_context': context_id,
        'knowledge_rules': ['R2', 'R3', 'R4', 'R5'],
        'conservative_heuristic': conservative(old, new),
        'polarity_from': p[old]['polarity'], 'polarity_to': p[new]['polarity'],
        'charge_from': p[old]['charge'], 'charge_to': p[new]['charge'],
        'size_from': p[old]['size'], 'size_to': p[new]['size'],
        'delta_hydropathy': round(p[new]['hydropathy'] - p[old]['hydropathy'], 3),
        'other_background_n': len(history),
        'other_background_median_delta': round(median(deltas), 5) if deltas else None,
        'other_background_positive_fraction': (
            round(sum(x > 0 for x in deltas) / len(deltas), 3) if deltas else None),
        'other_background_examples': sorted(
            history, key=lambda r: -abs(r['delta_in_proposed_direction']))[:max_examples],
        'epistasis_n': len(epistasis_history),
        'epistasis_same_fixed_background_n': sum(
            r['same_fixed_background'] for r in epistasis_history),
        'epistasis_examples': epistasis_history[:max_examples],
        'evidence_scope': 'KG0_or_earlier_rounds_measured_only',
        'is_candidate_fitness_known': False,
    }


def verified_evidence_text(evidence):
    """只陈述真实测量及其限定背景，不把上位性外推为候选的确定效应。"""
    parts = [f"提议替换 {evidence['substitution']}；其他背景配对数={evidence['other_background_n']}。"]
    if evidence['other_background_median_delta'] is not None:
        parts.append(f"其他背景中位Δ={evidence['other_background_median_delta']:+.5f}，不能外推当前候选真实Fitness。")
    for row in evidence['other_background_examples']:
        differences = sequence_differences(
            row['reference_variant'], evidence['reference_variant'], evidence['changed_position'])
        positions = [item['position'] for item in differences]
        parts.append(
            f"[{row['evidence_id']}] {row['reference_variant']}→{row['test_variant']}："
            f"起点Fitness={row['reference_fitness']:.7g}，终点Fitness={row['test_fitness']:.7g}，"
            f"Δ={row['delta_in_proposed_direction']:+.5f}。"
            f"历史参照{row['reference_variant']}与当前亲本{evidence['reference_variant']}"
            f"的固定背景差异位点={positions}；多位背景差异不能单独归因于其中一位。")
    parts.append(f"四角已测上位性对照数={evidence['epistasis_n']}，"
                 f"其中固定两位背景匹配数={evidence['epistasis_same_fixed_background_n']}。")
    for row in evidence['epistasis_examples']:
        parts.append(
            f"[{row['evidence_id']}] 联合替换{row['other_substitution']}，"
            f"背景={row['two_site_background']}，"
            f"按{evidence['substitution']}方向ε={row['epsilon_in_proposed_direction']:+.5f}（原始Fitness尺度）；"
            f"不代表候选的已知Fitness。")
    return ''.join(parts)


def knowledge_suggestions(anchor, eligible, scores, knowledge, paths, limit=8):
    """基于理化对照、真实差值及上位性证据产生可测试候选；不改 M0 分数。"""
    options = []
    for i in range(4):
        for new in AA:
            if new == anchor[i]:
                continue
            v = anchor[:i] + new + anchor[i + 1:]
            if v not in eligible:
                continue
            evidence = evaluate_substitution(knowledge, paths, anchor, v, max_examples=2)
            options.append({'variant': v, 'predicted_fitness': float(scores[v]),
                            'changed_position': POSITIONS[i], 'evidence': evidence})
    chosen = []
    present = set()

    def keep(item):
        if item['variant'] not in present and len(chosen) < limit:
            chosen.append(item)
            present.add(item['variant'])

    for flag in (True, False):
        subset = [x for x in options if x['evidence']['conservative_heuristic'] == flag]
        if subset:
            keep(max(subset, key=lambda x: (x['predicted_fitness'], x['variant'])))

    # 四角上位性用于优先提出有可检验对照的候选，不将 epsilon 加到预测 Fitness。
    epi_supported = [x for x in options if x['evidence']['epistasis_n']]
    for x in sorted(epi_supported, key=lambda r: (
            -int(r['evidence']['epistasis_same_fixed_background_n'] > 0),
            -r['evidence']['epistasis_same_fixed_background_n'],
            -r['predicted_fitness'], r['variant']))[:3]:
        keep(x)

    supported = [x for x in options if x['evidence']['other_background_n'] >= 2]
    for x in sorted(supported, key=lambda r: (
            -(r['evidence']['other_background_median_delta'] or 0),
            -r['predicted_fitness'], r['variant'])):
        keep(x)
    for x in sorted(options, key=lambda r: (-r['predicted_fitness'], r['variant'])):
        keep(x)
    return chosen


def knowledge_pairs(anchors, eligible, scores, knowledge, paths, existing, per_anchor=5):
    keys = {(p['test_variant'], p['reference_variant']) for p in existing}
    result = []
    for anchor in anchors:
        added = 0
        for row in knowledge_suggestions(anchor, eligible, scores, knowledge, paths, limit=12):
            v = row['variant']
            if (v, anchor) in keys:
                continue
            result.append({
                'test_variant': v, 'reference_variant': anchor,
                'changed_position': row['changed_position'],
                'reference_role': 'measured_reference',
                'selection_basis': 'kg_chemistry_and_measured_pair_query',
                'knowledge_evidence': row['evidence'],
            })
            keys.add((v, anchor))
            added += 1
            if added >= per_anchor:
                break
    return result


def focused_knowledge(knowledge, max_pairs=5):
    return {
        'type': knowledge['type'],
        'target_round': knowledge['target_round'],
        'rules': knowledge['rules'],
        'conservative_groups_heuristic': knowledge['conservative_groups_heuristic'],
        'observed_variant_count': knowledge['observed_variant_count'],
        'observed_one_site_pairs': knowledge['observed_one_site_pairs'],
        'observed_directed_pairs': knowledge['observed_directed_pairs'],
        'observed_mutation_combinations': knowledge['observed_mutation_combinations'],
        'observed_epistasis_interactions': knowledge['observed_epistasis_interactions'],
        'largest_measured_positive_pairs': knowledge['largest_measured_positive_pairs'][:max_pairs],
        'largest_measured_negative_pairs': knowledge['largest_measured_negative_pairs'][:max_pairs],
        'largest_absolute_epistasis': knowledge['largest_absolute_epistasis'][:max_pairs],
        'evidence_policy': knowledge['evidence_policy'],
    }


def kg_combination_proposals(anchors, eligible, scores, paths, per_anchor=12):
    """KG evidence generates NEW two-site candidates; none receive invented Fitness.

    Four-corner interactions transfer a substitution pattern across backgrounds.
    WT-relative mutation combinations provide weaker, context-limited patterns.
    All proposed variants must be in the frozen M0 pool and not yet measured.
    """
    from search_strategy import hamming

    epistasis = load_pairs(str(paths.folder / 'kg_observed_epistasis.csv'))
    combinations_df = load_pairs(str(paths.folder / 'kg_observed_combinations.csv'))
    positions = {p: i for i, p in enumerate(POSITIONS)}
    results = []
    for anchor in anchors:
        proposals = {}

        def propose(variant, evidence_id, evidence_type, context_match, details):
            if variant not in eligible or hamming(anchor, variant) != 2:
                return
            item = {
                'anchor_variant': anchor, 'variant': variant, 'evidence_id': evidence_id,
                'evidence_type': evidence_type, 'context_match': bool(context_match),
                'predicted_fitness': float(scores[variant]),
                'changed_positions': '+'.join(str(POSITIONS[i]) for i in range(4)
                                              if anchor[i] != variant[i]),
                'evidence_details': details,
            }
            prev = proposals.get(variant)
            rank = (int(evidence_type == 'four_corner_epistasis'),
                    int(context_match), item['predicted_fitness'])
            if prev is None or rank > (int(prev['evidence_type'] == 'four_corner_epistasis'),
                                        int(prev['context_match']), prev['predicted_fitness']):
                proposals[variant] = item

        for row in epistasis:
            i, j = positions[int(row['position_a'])], positions[int(row['position_b'])]
            first, second = row['substitution_a'], row['substitution_b']
            states_i, states_j = (first[0], first[-1]), (second[0], second[-1])
            if anchor[i] not in states_i or anchor[j] not in states_j:
                continue
            candidate = list(anchor)
            candidate[i] = states_i[1] if anchor[i] == states_i[0] else states_i[0]
            candidate[j] = states_j[1] if anchor[j] == states_j[0] else states_j[0]
            background = ''.join(anchor[k] for k in range(4) if k not in (i, j))
            match = background == row['background']
            evidence_id = (f"epistasis:{row['position_a']}-{row['position_b']}:"
                           f"{row['background']}:{states_i[0]}{states_i[1]}:"
                           f"{states_j[0]}{states_j[1]}")
            propose(''.join(candidate), evidence_id, 'four_corner_epistasis', match,
                    f"四角实测ε={float(row['epsilon']):+.4f};原背景={row['background']};"
                    '仅表示历史交互，不是新候选实测效应')

        for row in combinations_df:
            if int(row['mutation_count']) != 2:
                continue
            candidate = list(anchor)
            compatible = True
            for mutation in row['mutations'].split('+'):
                old, new, pos = mutation[0], mutation[-1], int(mutation[1:-1])
                i = positions[pos]
                if anchor[i] != old:
                    compatible = False
                    break
                candidate[i] = new
            if not compatible:
                continue
            evidence_id = f"{row['combination_id']}:source:{row['variant']}"
            propose(''.join(candidate), evidence_id, 'wt_relative_combination',
                    False, 'WT相对的历史共现组合；来源完整变体=' + row['variant'] +
                    '；其他突变=' + str(row['other_mutations']) +
                    '；来源变体Fitness不等于该子组合效应')

        # Prioritize real 2x2 evidence, but keep good M0 candidates and distinct site-pairs.
        ordered = sorted(proposals.values(), key=lambda r: (
            -int(r['evidence_type'] == 'four_corner_epistasis'),
            -int(r['context_match']), -r['predicted_fitness'], r['variant']))
        picked, site_pairs = [], set()
        for row in ordered:
            if row['changed_positions'] not in site_pairs:
                picked.append(row)
                site_pairs.add(row['changed_positions'])
            if len(picked) >= per_anchor:
                break
        for row in ordered:
            if len(picked) >= per_anchor:
                break
            if row['variant'] not in {x['variant'] for x in picked}:
                picked.append(row)
        results.extend(picked)
    return results
