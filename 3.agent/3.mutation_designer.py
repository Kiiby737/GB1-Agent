"""Identical proposal permissions and budgets; KG is ON-only decision evidence."""
import json
import pandas as pd
from agent_paths import get_round_arg, read_observed
from llm_client import generate_json
from search_strategy import search_state, validate_candidates
from common_search import candidate_pool, distance, POSITIONS, MAX_NEW_CONTROLS

N_PER_HYPOTHESIS = 20
MAX_BATCHES = 5
PROMPT = '''你是 GB1 Mutation Designer。四个字符对应39、40、41、54位。
两组都可从共同合法范围提出单点和双位点候选，每条假设目标20个设计候选。
根据候选的 M0预测、已测历史与当前假设作判断；预测不是实测结果。
每批最多提出 {n} 条，variant 必须来自 legal_variant_ids，排除已经提议或已测的ID。
candidate_local_neighbors仅为M0建议短名单；也可选择完整合法ID列表中的其他序列，不编造未知预测值。
单点/双位点设计目标由 remaining_by_distance 给定。双位点不能声称是严格单点配对。
reference_variant 可为空；非空时必须是已测且与候选只差一位的参照。
仅当输入存在knowledge时使用KG证据；KG历史交互不能当作新候选真实fitness。
输出严格JSON：{{"candidates":[{{"variant":"VWGM","hypothesis_id":"H1",
"purpose":"test","reference_variant":"","reason":"简短说明"}}]}}。
purpose为test或control；已测参照不申请新实验。'''


def main():
    p = get_round_arg()
    analysis = json.loads(p.analysis.read_text(encoding='utf-8'))
    hypotheses = json.loads(p.hypotheses.read_text(encoding='utf-8'))['hypotheses']
    state = search_state(p)
    if analysis['search_state'] != state:
        raise ValueError('Stale common search state')
    observed = read_observed(p)
    measured = set(observed.variant)
    full = candidate_pool(pd.read_csv(p.predictions), state['parent_variants'], measured,
                          set(pd.read_csv(p.test).variant))
    full.to_csv(p.folder/'common_candidate_pool.csv', index=False)
    records = full.set_index('variant').to_dict('index')
    proposals, proposed = [], set()
    knowledge = analysis.get('knowledge')
    combo_evidence = {}
    if knowledge is not None:
        from kg_agent_knowledge import kg_combination_proposals
        tips = kg_combination_proposals(state['parent_variants'], set(records),
                                       full.set_index('variant').predicted_fitness.to_dict(), p)
        tips = [r for r in tips if records[r['variant']]['anchor_variant']==r['anchor_variant']
                and int(records[r['variant']]['mutation_distance_to_anchor'])==2]
        pd.DataFrame(tips, columns=['anchor_variant','variant','evidence_id','evidence_type',
                     'context_match','predicted_fitness','changed_positions','evidence_details']).to_csv(
                         p.folder/'kg_combination_proposals.csv', index=False)
        combo_evidence = {(r['anchor_variant'], r['variant']):r for r in tips}
    targets = {1:20-state['two_site_target'], 2:state['two_site_target']}

    def add(v, h, method, purpose='test', reference='', reason='', enforce_target=False):
        v, reference = str(v).strip().upper(), str(reference or '').strip().upper()
        if v not in records or v in proposed or distance(v,h['anchor_variant']) not in (1,2):
            return False
        r = records[v]
        order = int(r['mutation_distance_to_anchor'])
        group = [x for x in proposals if x['hypothesis_id']==h['id']]
        if enforce_target and sum(x['mutation_distance_to_anchor']==order for x in group)>=targets[order]:
            return False
        if reference not in measured or distance(v,reference)!=1:
            reference = r['anchor_variant'] if order==1 else ''
        if purpose not in ('test','control'):
            return False
        if purpose=='control' and (not reference or sum(x['purpose']=='control' for x in proposals)>=MAX_NEW_CONTROLS):
            return False
        row = {k:value for k,value in r.items() if k not in ('predicted_fitness','uncertainty')}
        row.update(variant=v,hypothesis_id=h['id'],strategy=h['strategy'],purpose=purpose,
                   design_method=method,reference_variant=reference,reason=str(reason)[:80],
                   changed_position=(POSITIONS[next(i for i in range(4) if v[i]!=reference[i])]
                                     if reference else ''),
                   kg_evidence_id='',kg_evidence_type='',kg_context_match='',kg_evidence_details='')
        if knowledge is not None:
            from kg_agent_knowledge import evaluate_substitution
            evidence = evaluate_substitution(knowledge,p,reference or r['anchor_variant'],v)
            row.update(kg_substitution=evidence['substitution'] if evidence else '',
                       kg_conservative=evidence['conservative_heuristic'] if evidence else '',
                       kg_other_background_n=evidence['other_background_n'] if evidence else 0,
                       kg_other_background_median_delta=evidence['other_background_median_delta'] if evidence else '',
                       kg_evidence_reference=reference if evidence else '')
            item = combo_evidence.get((r['anchor_variant'],v))
            if item:
                row.update(kg_evidence_id=item['evidence_id'],kg_evidence_type=item['evidence_type'],
                           kg_context_match=item['context_match'],kg_evidence_details=item['evidence_details'])
        proposals.append(row)
        proposed.add(v)
        return True

    for h in hypotheses:
        pair = h['matched_comparison']
        if pair['reference_variant'] != h['anchor_variant'] or distance(pair['test_variant'],pair['reference_variant'])!=1:
            raise ValueError('Mandatory hypothesis comparison must use its measured parent')
        if pair['test_variant'] not in proposed:
            assert add(pair['test_variant'],h,'hypothesis_pair',reference=pair['reference_variant'],reason='必选单点比较')

    for h in hypotheses:
        local = full.loc[full.variant.map(lambda v: distance(v,h['anchor_variant']) in (1,2))]
        # Both modes receive the same M0-based shortlist construction.
        shortlist = pd.concat([local.loc[local.mutation_distance_to_anchor.eq(d)].head(24) for d in (1,2)])
        for batch in range(MAX_BATCHES):
            group = [x for x in proposals if x['hypothesis_id']==h['id']]
            if len(group)>=N_PER_HYPOTHESIS:
                break
            remaining = {str(d):max(0,targets[d]-sum(x['mutation_distance_to_anchor']==d for x in group)) for d in (1,2)}
            payload = {'hypothesis':h,'anchor':{'variant':h['anchor_variant'],
                       'measured_fitness':float(observed.set_index('variant').loc[h['anchor_variant'],'fitness'])},
                       'candidate_local_neighbors':shortlist.loc[~shortlist.variant.isin(proposed)].to_dict('records'),
                       'legal_variant_ids':local.variant.tolist(),
                       'remaining_by_distance':remaining,'already_proposed':sorted(proposed),
                       'observed':analysis['observed'],'last_round':analysis['last_round']}
            if knowledge is not None:
                from kg_agent_knowledge import focused_knowledge
                payload['knowledge'] = focused_knowledge(knowledge)
                payload['knowledge_design'] = h.get('knowledge_design')
                payload['kg_suggestions'] = [r for r in combo_evidence.values()
                                             if r['anchor_variant']==h['anchor_variant'] and r['variant'] in set(local.variant)]
            response = generate_json(PROMPT.format(n=min(10,20-len(group))),payload,
                                     debug_file=p.folder/f'invalid_mutation_{h["id"]}_{batch}.txt')
            for item in response.get('candidates',[]):
                if (item.get('hypothesis_id')==h['id'] and item.get('variant') in set(local.variant)
                        and len([x for x in proposals if x['hypothesis_id']==h['id']])<20):
                    add(item.get('variant',''),h,'llm',item.get('purpose','test'),
                        item.get('reference_variant',''),item.get('reason',''),True)
        for d in (2,1):
            for row in local.loc[local.mutation_distance_to_anchor.eq(d)].itertuples():
                if len([x for x in proposals if x['hypothesis_id']==h['id']])>=20:
                    break
                add(row.variant,h,'common_fallback',reason='共同范围内的M0补足',enforce_target=True)
    # Identical, KG-independent fallback guarantees the common batch allocation.
    for d in (1,2):
        for row in full.loc[full.mutation_distance_to_anchor.eq(d)].itertuples():
            if sum(x['mutation_distance_to_anchor']==d for x in proposals)>=targets[d]:
                break
            h = next(h for h in hypotheses if distance(row.variant,h['anchor_variant']) in (1,2))
            add(row.variant,h,'common_fallback',reason='共同实验配额补足')
    frame = pd.DataFrame(proposals)
    validate_candidates(p,frame,state)
    frame.to_csv(p.designed,index=False)
    print(f'Mutation Designer: {len(frame)} candidates; common single/two-site permissions; KG={knowledge is not None}')


if __name__=='__main__':
    main()
