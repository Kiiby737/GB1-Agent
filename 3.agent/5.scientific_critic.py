# -*- coding: utf-8 -*-
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from llm_client import generate_json
from agent_paths import get_round_arg, read_observed, kg_validation_slots
from search_strategy import search_state, validate_candidates, MAX_NEW_CONTROLS
from knowledge_validation import (
    validate_hypothesis_claims, validate_text_claims, text_fields, EPI_ID,
    epistasis_evidence_index, epistasis_reference_options, epistasis_reference_facts,
)

TOP_K = 20
REVIEW_BATCH_SIZE = 12  
KG_VALIDATION_SLOTS = 0  # No KG-exclusive allocations in this protocol.
POSITIONS = (39, 40, 41, 54)
AA = set('ACDEFGHIKLMNPQRSTVWY')

PROMPT = '''你是 GB1 定向进化 Agent 的 Scientific Critic。野生型四位点是 VDGV，variant 的四个字符分别为39、40、41、54位。
请审查【所有】候选与三条科学假设；你的判断仅用于科学解释，最终实验配额由 Python 确定性选择。

1. 明确区分：已测真实 fitness、固定 M0-AL predicted_fitness、uncertainty、自建 GB1-KG 的理化属性和历史已测单点配对。
2. candidate_facts 是 Python 验证过的位点和参照证据。只有其 comparison_status 为 measured_pair 或 candidate_pair 才能声称严格单点配对；unpaired 不得写成控制变量实验。
3. 已测参照可以用于比较，但不能作为新的实验候选，也不消耗新实验预算。
4. 如果 reference_variant 是另一条新候选，只有二者都被最终选择并查询 Oracle，才构成这轮完整的新实验配对。
5. 从上一轮真实实验分析每条假设受支持、受到挑战或仍缺乏证据的情况；不要用预测得分证明机制成立。
6. 单点差异不能独自证明两位点上位性；要检验协同需要在一致背景下设计2×2比较。
7. 多亲本候选的 anchor_variant 是真实已测亲本；双位点候选与该亲本相差两个位点，不是严格单点配对。须核实其历史上位性证据只在来源背景成立。即使某条候选与假设的预测方向矛盾，也可能具有证伪价值；指出其作用，而不是一律排除。
8. 禁止编造 Fitness、文献和位点；引用 knowledge 时区分通用性质和 GB1 已测证据，不得把理化性质等同于实测改进。
9. 历史证据必须核对 reference→test 的实际替换与 Δ=终点fitness−起点fitness。不同起始氨基酸的替换不能当作同一替换；反向箭头必须同时反转Δ。发现错误必须指出，禁止沿用。
10. 背景差异必须按完整四位点ID逐位核实，使用输入 background_differences_to_anchor；不得遗漏位点。多位背景同时改变时不能把差值单独归因于一位。“不能外推”不能抵消同句中错误的实测方向陈述；发现事实错误必须纠正后再输出。
11. 上位性引用须给出完整epistasis evidence_id，核对该记录的两个替换与当前被评估的目标替换是否一致。ε按目标替换方向引用：反转一个替换须反转ε符号；若引用规范方向须明确写“原始ε”。无关替换的记录不得作为目标替换的对应证据；不存在或缺少四角已测的记录不得引用。
12. 候选的epistasis_options与hypothesis_epistasis_options列出可引用的对应证据。优先在epistasis_evidence_ids数组填写完整ID，由Python核验并附上真实ε；scientific_reason、limitations、reason使用定性说明，不必手写ε数值。没有对应ID就填[]并说明缺少对应已测上位性证据，不得编写无来源ε数值。正文长度限制不包含ID数组。

仅输出 JSON：
{
  "hypothesis_assessments": [
    {"hypothesis_id":"H1","assessment":"模型支持|证据不足|模型不支持","reason":"综合评价",
     "epistasis_evidence_ids":[]}
  ],
  "reviews": [
    {"variant":"LYAG","hypothesis_alignment":"符合|不确定|矛盾",
     "scientific_reason":"解释设计依据及证据","limitations":"证据缺口与必要对照",
     "epistasis_evidence_ids":[]}
  ]
}
必须为输入的每条假设输出一条 assessment，且为每条 candidate 输出一条 review。
'''


def single_changed_position(a, b):
    if len(a) != 4 or len(b) != 4 or not set(a + b) <= AA:
        return None
    changed = [i for i in range(4) if a[i] != b[i]]
    return POSITIONS[changed[0]] if len(changed) == 1 else None


def prepare_candidates(scored, hypotheses):
    frame = scored.copy()
    frame['variant'] = frame['variant'].astype(str).str.upper().str.strip()
    assert frame['variant'].is_unique, 'Duplicate variant in scored_candidates.csv'
    for col in ('reference_variant', 'strategy', 'anchor_variant', 'design_method'):
        if col not in frame:
            frame[col] = ''
        frame[col] = frame[col].fillna('').astype(str).str.strip()
    frame['reference_variant'] = frame['reference_variant'].str.upper()
    frame['anchor_variant'] = frame['anchor_variant'].str.upper()
    strategies = {h['id']: h.get('strategy', 'mechanism_validation') for h in hypotheses}
    frame['strategy'] = frame.apply(
        lambda r: r['strategy'] or strategies[r['hypothesis_id']], axis=1)
    assert frame['purpose'].isin(['test', 'control']).all(), 'Invalid purpose'
    assert frame['variant'].map(lambda v: len(v) == 4 and set(v) <= AA).all()
    return frame.sort_values('predicted_fitness', ascending=False).reset_index(drop=True)


def build_pair_facts(scored, observed):
    """All comparisons use exact 1-site differences; observed excludes the hidden Oracle."""
    fitness = dict(zip(observed['variant'].astype(str).str.upper(), observed['fitness']))
    rows = scored.set_index('variant').to_dict('index')
    facts = {}
    for variant, row in rows.items():
        ref = row['reference_variant']
        position = single_changed_position(variant, ref)
        if position is not None and ref in fitness:
            status = 'measured_pair'
        elif position is not None and ref in rows:
            status = 'candidate_pair'
        else:
            status = 'unpaired'
        facts[variant] = {
            'design_method': row.get('design_method', ''),
            'kg_evidence_id': row.get('kg_evidence_id', ''),
            'parent_distance': sum(a != b for a, b in zip(variant, row['anchor_variant'])),
            'variant': variant,
            'site_39_40_41_54': {str(pos): aa for pos, aa in zip(POSITIONS, variant)},
            'mutations_from_wt': [f'{wt}{pos}{aa}' for wt, pos, aa in zip('VDGV', POSITIONS, variant) if wt != aa],
            'reference_variant': ref if status != 'unpaired' else None,
            'comparison_status': status,
            'changed_position': position if status != 'unpaired' else None,
            'reference_measured_fitness': float(fitness[ref]) if status == 'measured_pair' else None,
        }
    return facts


def selection_priority(scored, knowledge=None):
    """Identical frozen M0-AL prediction ranking in ON and OFF, no KG pseudo-fitness."""
    ordered = scored.sort_values(['predicted_fitness', 'variant'], ascending=[False, True])
    n = len(ordered)
    return {v: (n - i) / n for i, v in enumerate(ordered['variant'])}


def select_topk(scored, observed, hypotheses, top_k=TOP_K, knowledge=None, parent_variant=None,
                reserved_kg_slots=0, parent_variants=None):
    from common_search import select_batch, distance
    if reserved_kg_slots:
        raise ValueError('ON-only reserved slots are disabled in the common-search protocol')
    df = prepare_candidates(scored, hypotheses)
    if 'mutation_distance_to_anchor' not in df:
        df['mutation_distance_to_anchor'] = df.apply(lambda r: distance(r.variant,r.anchor_variant),axis=1)
    parents = parent_variants or ([parent_variant] if parent_variant else [])
    parents = list(dict.fromkeys(parents + sorted(set(df.anchor_variant)-set(parents))))
    facts = build_pair_facts(df, observed)
    required = []
    known = set(observed.variant)
    for h in hypotheses:
        pair = h['matched_comparison']
        v, ref = pair['test_variant'], pair['reference_variant']
        if ref not in known or single_changed_position(v,ref) is None:
            raise ValueError('Mandatory comparisons require a measured one-site reference')
        row = df.loc[df.variant.eq(v)]
        if len(row)!=1 or row.iloc[0].design_method!='hypothesis_pair':
            raise ValueError('Mandatory hypothesis test was not preserved')
        required.append(v)
    picked, targets = select_batch(df,known,parents,required=required,top_k=top_k)
    picked['reserved_kg_slots'] = 0
    picked['selection_priority'] = picked.variant.map(selection_priority(df))
    picked['two_site_target'] = targets[2]
    for i,row in picked.iterrows():
        fact = facts[row.variant]
        ref = fact['reference_variant']
        picked.at[i,'pair_type'] = 'measured_reference_pair' if fact['comparison_status']=='measured_pair' else 'unpaired_exploration'
        picked.at[i,'effective_reference_variant'] = ref or ''
        picked.at[i,'verified_changed_position'] = single_changed_position(row.variant,ref) if ref else ''
    picked = picked.sort_values(['predicted_fitness','variant'],ascending=[False,True]).reset_index(drop=True)
    picked = picked.drop(columns=['prediction_rank'],errors='ignore')
    picked.insert(0,'prediction_rank',range(1,len(picked)+1))
    return picked,facts


def add_epistasis_options(payload, observed):
    """Offer evidence for each actual review target, not just its hypothesis."""
    index = epistasis_evidence_index(observed)
    payload['hypothesis_epistasis_options'] = {
        h['id']:epistasis_reference_options(h['matched_comparison']['reference_variant'],
                                            h['matched_comparison']['test_variant'],observed,index=index)
        for h in payload['hypotheses']}
    for row in payload['candidates']:
        row['epistasis_options'] = epistasis_reference_options(
            row['anchor_variant'],row['variant'],observed,index=index)


def attach_verified_epistasis_references(result, payload, observed):
    hypotheses = {h['id']:h for h in payload.get('hypotheses',())}
    candidates = {r['variant']:r for r in payload.get('candidates',())}
    index = None
    for field in ('hypothesis_assessments','reviews'):
        for item in result.get(field,[]):
            if not isinstance(item,dict):
                raise ValueError(f'{field} entries must be objects')
            ids = list(item.get('epistasis_evidence_ids',[]))
            ids += [eid for _,text in text_fields(item) for eid in EPI_ID.findall(text)]
            ids = list(dict.fromkeys(ids))
            item['epistasis_evidence_ids'] = ids
            if not ids:
                item['verified_epistasis_evidence'] = []
                continue
            if index is None:
                index = epistasis_evidence_index(observed)
            if field == 'hypothesis_assessments':
                pair = hypotheses.get(item.get('hypothesis_id'),{}).get('matched_comparison',{})
                reference,test = pair.get('reference_variant'),pair.get('test_variant')
            else:
                row = candidates.get(item.get('variant'),{})
                reference,test = row.get('anchor_variant'),row.get('variant')
            item['verified_epistasis_evidence'] = [
                epistasis_reference_facts(eid,reference,test,observed,index=index) for eid in ids]


def generate_verified_review(prompt, payload, observed, debug_file):
    verified_prompt = prompt
    verified_payload = payload
    trace_id = time.time_ns()
    for attempt in range(3):
        result = generate_json(verified_prompt, verified_payload, debug_file=debug_file)
        try:
            validate_text_claims(result, observed, hypotheses=payload.get('hypotheses', ()),
                                 candidates=payload.get('candidates', ()))
            attach_verified_epistasis_references(result,payload,observed)
            validate_text_claims(result, observed, hypotheses=payload.get('hypotheses', ()),
                                 candidates=payload.get('candidates', ()))
            return result
        except ValueError as exc:
            if debug_file is not None:
                path = Path(debug_file)
                path.parent.mkdir(parents=True,exist_ok=True)
                path = path.with_name(f'{path.stem}_factual_{trace_id}_attempt{attempt+1}.json')
                path.write_text(json.dumps({'attempt':attempt+1,'batch':payload.get('batch'),
                                            'error':str(exc),'response':result},ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'Critic factual check rejected attempt {attempt+1}/3: {exc}',flush=True)
            if attempt == 2:
                raise ValueError(f'Critic factual check failed after 3 attempts: {exc}') from exc
            verified_prompt = (prompt + '\nPython事实核查未通过，请纠正后重新输出：' + str(exc)
                + '\n核对factual_correction.invalid_reply中的具体错误。保留所有候选和假设评价。'
                  '上位性引用优先填写epistasis_evidence_ids数组中的完整ID，数值由Python附加；'
                  '若无对应证据，改为不含ε数值的定性说明并填[]，不要重复无来源的数值。')
            verified_payload = {**payload,'factual_correction':{'error':str(exc),'invalid_reply':result}}


def main():
    p = get_round_arg()
    raw_scored = pd.read_csv(p.scored)
    state = search_state(p)
    validate_candidates(p, raw_scored, state)
    hypotheses_json = json.loads(p.hypotheses.read_text(encoding='utf-8'))
    hypotheses = hypotheses_json['hypotheses']
    analysis = json.loads(p.analysis.read_text(encoding='utf-8'))
    observed = read_observed(p)
    # Concrete evidence errors block selection, independently of LLM agreement.
    validate_hypothesis_claims(hypotheses, observed)
    scored = prepare_candidates(raw_scored, hypotheses)
    facts = build_pair_facts(scored, observed)


    columns = ['variant', 'hypothesis_id', 'purpose', 'reason', 'strategy',
               'anchor_variant', 'reference_variant', 'predicted_fitness', 'uncertainty']
    if 'knowledge' in analysis:
        columns += [c for c in (
            'kg_substitution', 'kg_conservative', 'kg_other_background_n',
            'kg_evidence_id', 'kg_evidence_type', 'kg_context_match',
            'mutation_distance_to_anchor', 'kg_evidence_details',
            'kg_other_background_median_delta', 'kg_evidence_reference', 'kg_rule_ids',
        ) if c in scored.columns]
    payload = {
        'hypotheses': hypotheses,
        'candidates': scored[columns].to_dict('records'),
        'candidate_facts': list(facts.values()),
        'observed': analysis['observed'],
        'last_round': analysis['last_round'],
        'predicted_summary': {
            'count': analysis['predicted']['count'],
            'top100_site_stats': analysis['predicted']['top100_site_stats'],
            'top100_vs_pool': analysis['predicted']['top100_vs_pool'],
        },
    }
    if 'knowledge' in analysis:
        from kg_agent_knowledge import focused_knowledge
        payload['knowledge'] = focused_knowledge(analysis['knowledge'])
        payload['knowledge_selection_policy'] = (
            'KG informs hypothesis and mutation design. Fixed M0-only prediction rank, '
            'mandatory matched tests and scientific controls determine the final Top-20; '
            'same single/two-site quota in both modes; '
            'no artificial Fitness bonus is added.')
        payload['knowledge_design_reviews'] = {
            h['id']: h['knowledge_design'] for h in hypotheses
            if 'knowledge_design' in h
        }
        add_epistasis_options(payload,observed)


    all_reviews = []
    hypothesis_assessments = None
    candidate_rows = payload['candidates']
    for offset in range(0, len(candidate_rows), REVIEW_BATCH_SIZE):
        batch_number = offset // REVIEW_BATCH_SIZE + 1
        batch_candidates = candidate_rows[offset:offset + REVIEW_BATCH_SIZE]
        batch_variants = [r['variant'] for r in batch_candidates]
        batch_payload = dict(payload)
        batch_payload['candidates'] = batch_candidates
        batch_payload['candidate_facts'] = [facts[v] for v in batch_variants]
        batch_payload['batch'] = {
            'number': batch_number,
            'total_candidates_this_batch': len(batch_candidates),
            'total_candidates_all_batches': len(candidate_rows),
        }
        if batch_number == 1:
            batch_prompt = (
                PROMPT + '\n本次只审查输入的 candidates 中的候选（不是其他批次），'
                'hypothesis_assessments 必须包含全部三条假设。'
                'reviews 中每个候选只出现一次；理由与局限分别不超过50汉字，ID单独填数组。'
            )
        else:
            batch_prompt = (
                PROMPT + '\n这不是第一批：本次只输出 '
                '{"reviews": [{"variant": "...", "hypothesis_alignment": "符合|不确定|矛盾", '
                '"scientific_reason": "...", "limitations": "...", "epistasis_evidence_ids": []}]}。'
                '禁止输出 hypothesis_assessments，且只审查本批 candidates。'
                'reviews 中每个候选只出现一次；理由与局限分别不超过50汉字，ID单独填数组。'
            )
        batch_review = generate_verified_review(
            batch_prompt, batch_payload, observed,
            p.folder / f'invalid_critic_batch_{batch_number:02d}.txt',
        )
        batch_reviews = batch_review['reviews']
        returned_variants = [r['variant'] for r in batch_reviews]
        assert len(returned_variants) == len(batch_variants) and set(returned_variants) == set(batch_variants), (
            f'Critic batch {batch_number}: review variants differ from the requested batch'
        )
        for item in batch_reviews:
            assert all(field in item for field in
                       ('variant', 'hypothesis_alignment', 'scientific_reason', 'limitations')), (
                f'Critic batch {batch_number}: missing review fields'
            )
        all_reviews.extend(batch_reviews)
        if batch_number == 1:
            hypothesis_assessments = batch_review['hypothesis_assessments']
        print(f'Scientific Critic batch {batch_number}: reviewed '
              f'{len(batch_reviews)} / {len(candidate_rows)} candidates '
              f'(cumulative {len(all_reviews)})', flush=True)

    review = {'hypothesis_assessments': hypothesis_assessments, 'reviews': all_reviews}
    reviewed_variants = [r['variant'] for r in review['reviews']]
    assessment_ids = [r['hypothesis_id'] for r in review['hypothesis_assessments']]
    assert len(reviewed_variants) == len(scored) and set(reviewed_variants) == set(scored['variant'])
    assert len(assessment_ids) == len(hypotheses) and set(assessment_ids) == {h['id'] for h in hypotheses}

    slots = kg_validation_slots()
    picked, _ = select_topk(scored, observed, hypotheses, knowledge=analysis.get('knowledge'),
                          parent_variant=state['parent_variant'], reserved_kg_slots=slots, parent_variants=state['parent_variants'])
    validate_candidates(p, picked, state)
    review_fields = pd.DataFrame(review['reviews'])[
        ['variant', 'hypothesis_alignment', 'scientific_reason', 'limitations']
    ]
    picked = picked.merge(review_fields, on='variant', how='left', validate='one_to_one')
    orphaned = picked['pair_type'].eq('unpaired_exploration') & picked['reference_variant'].ne('')
    picked.loc[orphaned, 'limitations'] = (
        picked.loc[orphaned, 'limitations'].fillna('') +
        '；设计时参照序列未作为有效本轮配对入选，不得据此推断单点效应'
    )
    test_counts = Counter(picked.loc[picked['purpose'] == 'test', 'hypothesis_id'])
    picked_by_variant = picked.set_index('variant')
    for h in hypotheses:
        test = h['matched_comparison']['test_variant']
        if picked_by_variant.loc[test, 'hypothesis_id'] != h['id']:
            test_counts[h['id']] += 1
    review['selection_audit'] = {
        'n_candidates_reviewed':len(scored), 'n_selected':len(picked),
        'n_test':int(picked.purpose.eq('test').sum()), 'n_control':int(picked.purpose.eq('control').sum()),
        'n_single_site':int(picked.mutation_distance_to_anchor.eq(1).sum()),
        'n_two_site':int(picked.mutation_distance_to_anchor.eq(2).sum()),
        'two_site_target':state['two_site_target'], 'parent_variants':state['parent_variants'],
        'hypothesis_test_counts':dict(test_counts),
        'mandatory_comparisons':{h['id']:h['matched_comparison'] for h in hypotheses},
        'pair_type_counts':dict(Counter(picked['pair_type'])),
        'admissible_pool_sha256':state['admissible_pool_sha256'],
        'policy':'Both modes: common radius-two domain, target 6 two-site tests, same fallback and control cap',
        'knowledge_policy':'ON-only evidence input; no KG-exclusive search permissions or reserved slots'
                           if 'knowledge' in analysis else 'OFF: no KG retrieval, graph, or KG-filtered proposals',
        'kg_evidence_selected':int(picked.get('kg_evidence_id',pd.Series('',index=picked.index)).fillna('').ne('').sum()),
    }
    if 'knowledge' in analysis:
        review['selection_audit']['knowledge_source']=analysis['knowledge']['source_graph']
    p.review.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    picked.to_csv(p.topk, index=False)
    print(f'Scientific Critic: reviewed {len(scored)} candidates; selected {len(picked)}')
    print(json.dumps(review['selection_audit'], ensure_ascii=False, indent=2))
    print(f'Scientific review: {p.review}')
    print(f'Top-{len(picked)} recommendations: {p.topk}')
    print(picked[['prediction_rank', 'variant', 'hypothesis_id', 'purpose',
                  'strategy', 'pair_type', 'predicted_fitness']].to_string(index=False))


if __name__ == '__main__':
    main()
