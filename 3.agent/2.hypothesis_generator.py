# -*- coding: utf-8 -*-
import json

import pandas as pd

from llm_client import generate_json
from agent_paths import get_round_arg, read_observed
from search_strategy import search_state, hamming
from knowledge_validation import validate_hypothesis_claims, validate_text_claims


WT = 'VDGV'
POSITIONS = (39, 40, 41, 54)
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


def kg_pair_evidence(knowledge, paths, reference, candidate):
    from kg_agent_knowledge import evaluate_substitution
    return evaluate_substitution(knowledge, paths, reference, candidate)



PROMPT = '''你是 GB1 蛋白定向进化科学智能体的 Hypothesis Generator。
野生型四位点 VDGV，variant 四个字母依次对应 39、40、41、54 位。
固定 M0-AL 模型不更新。你的任务是基于截至当前轮已经获得的真实实验数据、预测结果、上一轮反馈，以及可用的自建 GB1-KG 知识，提出可检验的下一轮突变假设。

目标有两个，必须区分：
A. 优化目标：提高下一轮新实验候选的真实 Fitness，尤其关注 Agent 上一轮已测高 Fitness 变体的邻域。
B. 科学目标：设计可解释、可证伪的比较，检验突变在特定序列背景下的贡献。
不要为了获得更高的预测 Fitness 而放弃有价值的实验对照；也不要为了凑满科学假设而完全偏离上一轮成功的变体。
M0-AL 的 996 条训练数据和独立初始20条实验均为两组共同可见的历史知识；还可使用本分支严格早于本轮的真实结果。996条与初始20条不属于本轮新发现，不能进入新实验候选，也不能计入每轮的20条新测结果。模型固定；禁止使用当前/未来轮 Oracle 标签。

【第一步：复盘上一轮】
- 如果 last_round 为 null，previous_hypotheses_review 输出空列表。
- 如果存在上一轮数据：必须对 previous_hypotheses 的每条假设分别复盘，并使用 agent_feedback 给出的分组真实结果。
- 逐条给出「支持倾向 / 受到挑战 / 证据不足」，以及具体实验变体及真实 Fitness。
- 如果某条假设没有 test，或者缺少严格配对，只能写「证据不足」，不可从 control 的高分推断其得到验证。
- 判断是否值得继续时，明确是局部优化、修正假设、还是停止沿用该机制；不能把不同背景的边际均值差当成单点突变因果效应。
- 必须对照 previous_scientific_feedback：上一轮 Scientific Critic 的 assessment、科学理由及局限是在 Oracle 揭示前写出的，不是实测结论。
- 每个 previous_hypotheses_review 都要给出 critic_reflection：结合此前 Critic 关注的风险、上一轮新测序列的真实 Fitness 和已测单点配对差值，说明哪些预先判断与实验一致、哪些受挑战、哪些依然无法检验。
- previous_review_facts 给出Python核算的上一轮真实配对Δ与当时KG跨背景中位Δ，二者是不同指标。复盘须分别说明“当前配对实测Δ”与“上一轮KG跨背景中位Δ”，建议分句；不得把历史中位Δ写成当前配对Δ，也不得把Critic实验前的弱证据当作实验后的结论。
- 上一轮 Critic 的「模型支持」绝不等于「实验支持」。若缺少同背景真实配对，不得把高分或均值差写成机制验证；没有充分证据时坚持「证据不足」。
- 新一轮的 hypothesis / design_guidance / falsification_test 必须回应上一轮尚未解决的 Critic 局限，或明确说明为何改换新的研究方向。优先安排能解决旧问题的有效对照。

【第二步：提出恰好三条本轮假设】
- H1/H2/H3 的 anchor_variant 必须分别等于输入 hypothesis_anchors 对应的实测亲本；matched_comparison.reference_variant 必须等于相应 anchor。
- Round 1 唯一亲本为指定初始序列；后续同时保留最多三个初始序列的已测后代亲本，不能选择 M0 或随机初始20条中的全局高分序列作为新分支根。
- 每条假设的必选 matched_comparison 保留与自身亲本相差一位的真实可测比较；此外两组的 Mutation Designer 均可提出距离已测亲本两位的候选；只有 ON 可读取 KG 证据，不能将其说成单点实测配对。
- 不预设任何具体的有益突变，不强制三条假设属于不同固定类型；每条的 strategy 从 local_optimization、mechanism_validation、knowledge_exploration 中选择最符合的一项。
- 优先选择能够通过有限新实验验证的局部变化。距离上一轮高分序列较远的探索应说明其科学依据及风险。
- 仅在输入提供完整候选池背景统计时才谈相对富集；模型预测、突变频率、历史均值都不能被说成实验因果证明。
- 严格区分真实 fitness、预测 predicted_fitness、ensemble uncertainty 和知识图谱理化性质。知识图谱只能支持机制合理性，不能证明变体的 Fitness。
- 若提供 knowledge：查询自建 GB1-KG 的 R1-R7、氨基酸理化属性、保守/激进替换、历史单点配对、序列背景及四角已测上位性。历史改善只在已测背景成立，跨背景配对可作弱证据、不可当作本候选的实测结果；知识不能保证收益。
- 所有历史箭头必须按 reference_variant→test_variant 引用，Δ=终点fitness−起点fitness。
- 历史配对与当前亲本的背景差异必须逐位核对，优先引用 knowledge_evidence 的 background_differences_to_anchor。比较时写出完整四位点ID及全部差异位点；三字母背景必须说明省略的位点。若两个以上固定背景位点不同，不能把效应差异单独归因于其中一个位点。
- 引用上位性时必须保留完整 epistasis evidence_id，并核对目标替换与记录中的替换是否相同或反向；不同替换不能作为本假设的对应证据。ε统一按本假设目标替换方向引用 epsilon_in_proposed_direction；原始方向数值须明确写“原始ε”。不能只按位点相同就引用。缺少四角已测证据时不编造ID或ε数值。
- knowledge_rationale 和 observed_evidence 内的历史箭头只列与所提议突变完全相同的替换（位点、起始和终止氨基酸均相同）。其他替换的背景信息放在 design_guidance 等字段并注明不是同一替换；M40W/C40W/L40W不能写成D40W证据。

【第三步：提供可执行的单点比较设计】
- 每条假设给出 matched_comparison.pair_id（整数），必须从输入 feasible_measured_pairs 中已有的 pair_id 选择。Python 会还原 test_variant、reference_variant、changed_position、reference_role。不要自行拼写实验序列。
- 两条四氨基酸序列必须恰好有一处差异。changed_position 必须根据四字母的实际差异计算：第1/2/3/4个字母分别是39/40/41/54位。
- reference_role 为 measured_reference（已测参照，不能再申请新 Oracle 标签）或 new_control（拟新测对照）；不要在不确定是否已测的情况下声称其一定未测。
- 三条假设的 matched_comparison 可以共享同一条未测候选，同一 variant 只需实验一次。test_variant **必须未测且在当前候选池内**；reference_variant 必须是已测参照或另一条未测且可测的 candidate。已测-已测配对或 test_variant 已测的配对一律不合法。优先参考 feasible_measured_pairs（test 均可新测，reference 均为已测）和 feasible_local_neighbors，不要凭空指定不存在的候选。若 strategy=local_optimization，matched_comparison 中的新测成员须是 anchor 的单点邻居；额外设计候选仍按两组共同的半径二规则生成。
- 如假设关于两位点相互作用，单点配对本身不足以证明协同，必须在 falsification_test 说明需要相同背景下的 2×2 对照。
- 不编造变体的真实 Fitness、模型预测或文献结果；不能把无重复数据的差异称作统计显著。
- 只提出假设，不自行调用 Oracle，也不选择最终 Top20。

只返回如下 JSON 对象，保留示例中的全部假设字段；Knowledge ON 时两个 knowledge_* 字段均为必填：
{
  "previous_hypotheses_review": [
    {
      "previous_id": "H1",
      "evidence_status": "支持倾向|受到挑战|证据不足",
      "actual_results": "上一轮真实实验及其局限",
      "critic_reflection": "对照上一轮实验前Critic评价与真实结果，说明预测、局限、已验证配对和未解决问题",
      "next_action": "继续局部优化|修正方向|暂缓检验"
    }
  ],
  "hypotheses": [
    {
      "id": "H1",
      "strategy": "local_optimization|mechanism_validation|knowledge_exploration",
      "based_on_previous": "上一轮假设ID或null",
      "anchor_variant": "必须为对应 hypothesis_anchors 指定的亲本",
      "hypothesis": "可检验的突变假设，说明适用的序列背景和预期方向",
      "observed_evidence": "真实实验支持、相反证据、具体数值及局限",
      "predicted_evidence": "M0模型预测支持、矛盾和不确定性；若无具体预测则说明",
      "design_guidance": "下一轮建议从哪些已测高分变体的哪个局部邻域设计新候选，以及探索风险",
      "matched_comparison": {"pair_id": 0},
      "falsification_test": "什么真实实验结果会反驳该假设、需要哪些配对或2×2比较",
      "knowledge_rationale": "Knowledge ON时必填：结合输入KG证据说明本假设依据与背景限制；OFF可为空字符串",
      "knowledge_falsification": "Knowledge ON时必填：什么比较会挑战这条知识解释及其局限；OFF可为空字符串"
    }
  ],
  "summary": "前一轮发现、被反驳和未检验的模式、本轮探索与优化的权衡"
}
如果输入没有上一轮实验结果，previous_hypotheses_review 必须为 []。
matched_comparison 中的 pair_id=0 仅为格式示例，不是预设选择；必须从实际输入的合法配对编号中选择。
Knowledge ON 时，H1、H2、H3 每一条都必须在假设对象顶层包含 knowledge_rationale 和 knowledge_falsification，值为非空字符串；不能只在 summary 中提供，也不能省略或填 null。
'''


def required_knowledge_text(hypothesis, field):
    value = hypothesis.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{hypothesis.get('id', 'hypothesis')}: {field} must be a non-empty string "
            'at the top level of each hypothesis when Knowledge ON; '
            'include both knowledge_rationale and knowledge_falsification for H1/H2/H3')
    return value.strip()


def build_agent_feedback(data):
    last = data['last_round']
    if last is None:
        return None
    rows = last['results']
    feedback = []
    for hypothesis in data['previous_hypotheses']:
        matching = [r for r in rows if r['hypothesis_id'] == hypothesis['id']]
        tests = [r for r in matching if r['purpose'] == 'test']
        controls = [r for r in matching if r['purpose'] == 'control']
        feedback.append({
            'hypothesis_id': hypothesis['id'],
            'n_test': len(tests),
            'n_control': len(controls),
            'mean_test_fitness': round(sum(r['fitness'] for r in tests) / len(tests), 4) if tests else None,
            'mean_control_fitness': round(sum(r['fitness'] for r in controls) / len(controls), 4) if controls else None,
            'best_test': max(tests, key=lambda r: r['fitness']) if tests else None,
            'best_control': max(controls, key=lambda r: r['fitness']) if controls else None,
            'assessment_note': 'Means are descriptive, not matched causal effects; no test means no direct test.',
        })
    return {
        'last_round_top5': sorted(rows, key=lambda r: r['fitness'], reverse=True)[:5],
        'by_previous_hypothesis': feedback,
    }


def build_previous_review_facts(data, observed):
    """Keep the actual matched contrast separate from the prior KG statistic."""
    fitness = dict(zip(observed.variant, observed.fitness))
    facts = {}
    for h in data['previous_hypotheses']:
        pair = h['matched_comparison']
        reference, test = pair['reference_variant'], pair['test_variant']
        measured = reference in fitness and test in fitness and hamming(reference, test) == 1
        delta = float(fitness[test]) - float(fitness[reference]) if measured else None
        evidence = h.get('knowledge_design', {}).get('selected_pair_evidence', {})
        median = evidence.get('other_background_median_delta')
        text = (f'当前已测配对{reference}→{test}：实测Δ={delta:+.9f}。'
                if measured else f'{reference}→{test}尚无完整已测单点比较。')
        if median is not None:
            text += f'上一轮KG其他背景中位Δ={float(median):+.9f}，与当前配对实测Δ是不同指标。'
        facts[h['id']] = {
            'reference_variant': reference, 'test_variant': test,
            'reference_fitness': float(fitness[reference]) if reference in fitness else None,
            'test_fitness': float(fitness[test]) if test in fitness else None,
            'observed_single_site_delta': delta,
            'previous_KG_other_background_median_delta': median,
            'previous_KG_other_background_n': evidence.get('other_background_n'),
            'verified_fact_text': text,
        }
    return facts


def validate_matched_comparison(hypothesis):
    pair = hypothesis['matched_comparison']
    a = pair['test_variant']
    b = pair['reference_variant']
    assert len(a) == len(b) == 4 and set(a + b) <= AMINO_ACIDS, \
        f"{hypothesis['id']}: invalid variant in matched_comparison"
    different = [i for i in range(4) if a[i] != b[i]]
    assert len(different) == 1, \
        f"{hypothesis['id']}: pair must differ at exactly one site: {a} vs {b}"
    actual_position = POSITIONS[different[0]]
    if int(pair['changed_position']) != actual_position:
        print(f"{hypothesis['id']}: corrected changed_position "
              f"{pair['changed_position']} -> {actual_position} for {a}/{b}", flush=True)
        pair['changed_position'] = actual_position


def validate_round_plan(result, data, measured, available):
    hypotheses = result['hypotheses']
    assert len(hypotheses) == 3, 'Need exactly 3 hypotheses'
    assert [h['id'] for h in hypotheses] == ['H1', 'H2', 'H3'], 'IDs must be H1/H2/H3'

    for h in hypotheses:
        validate_matched_comparison(h)
        pair = h['matched_comparison']
        parent = data['hypothesis_anchors'][h['id']]
        if h.get('anchor_variant') != parent or pair['reference_variant'] != parent:
            raise ValueError(f"{h['id']}: selected comparison must match assigned measured parent {parent}")
        if hamming(parent, pair['test_variant']) != 1:
            raise ValueError('Required test is outside the current parent single-site neighborhood')
        test, reference = pair['test_variant'], pair['reference_variant']
        assert test in available, (
            f"{h['id']}: test_variant {test} must be an unmeasured candidate in the eligible pool"
        )
        assert reference in (measured | available), (
            f"{h['id']}: reference_variant {reference} is neither measured nor an eligible new control"
        )
        expected_role = 'measured_reference' if reference in measured else 'new_control'
        if pair.get('reference_role') != expected_role:
            print(f"{h['id']}: corrected reference_role to {expected_role}", flush=True)
            pair['reference_role'] = expected_role
        new_members = {test, reference} & available
        if h['strategy'] == 'local_optimization':
            anchor = h['anchor_variant']
            assert anchor in measured, f"{h['id']}: anchor {anchor} has no observed label"
            for variant in new_members:
                assert sum(x != y for x, y in zip(variant, anchor)) == 1, (
                    f"{h['id']}: new candidate/control {variant} is not a 1-site neighbor of anchor {anchor}"
                )

    if data['last_round'] is None:
        assert result['previous_hypotheses_review'] == [], 'Round 1 requires empty review'
    else:
        ids = [r['previous_id'] for r in result['previous_hypotheses_review']]
        prev = [h['id'] for h in data['previous_hypotheses']]
        assert sorted(ids) == sorted(prev), 'All previous hypotheses must be reviewed'
        for item in result['previous_hypotheses_review']:
            assert str(item.get('critic_reflection', '')).strip(), (
                f"{item['previous_id']}: missing comparison of prior Critic assessment with Oracle results"
            )


def main():
    p = get_round_arg()
    data = json.loads(p.analysis.read_text(encoding='utf-8'))
    state = search_state(p)
    if data.get('search_state') != state:
        raise ValueError('Data analysis has a stale or missing seeded-search state')
    data['agent_feedback'] = build_agent_feedback(data)

    observed = read_observed(p)
    data['previous_review_facts'] = build_previous_review_facts(data, observed)
    measured = set(observed['variant'])
    predictions = pd.read_csv(p.predictions, usecols=['variant', 'predicted_fitness'])
    predictions['variant'] = predictions['variant'].astype(str).str.upper().str.strip()
    held_out = set(pd.read_csv(p.test, usecols=['variant'])['variant'].str.upper().str.strip())
    available_scores = dict(zip(predictions['variant'], predictions['predicted_fitness']))
    available = set(available_scores) - measured - held_out
    
    anchors = state['parent_variants']
    data['hypothesis_anchors'] = {f'H{i}': anchors[min(i - 1, len(anchors) - 1)]
                                  for i in range(1, 4)}
    print(f"Seed {state['start_variant']} -> Round {p.round_num} parents {anchors}", flush=True)


    aa = 'ACDEFGHIKLMNPQRSTVWY'
    data['feasible_local_neighbors'] = {}
    for anchor in anchors:
        neighbors = [
            anchor[:i] + residue + anchor[i + 1:]
            for i in range(4) for residue in aa if residue != anchor[i]
        ]
        neighbors = [v for v in neighbors if v in available]
        data['feasible_local_neighbors'][anchor] = sorted(
            neighbors, key=lambda v: -available_scores[v]
        )[:15]

    data['feasible_measured_pairs'] = [
        {
            'test_variant': variant,
            'reference_variant': anchor,
            'changed_position': POSITIONS[next(
                i for i in range(4) if variant[i] != anchor[i]
            )],
            'reference_role': 'measured_reference',
        }
        for anchor in anchors
        for variant in data['feasible_local_neighbors'][anchor][:10]
    ]
    if 'knowledge' in data:
        from kg_agent_knowledge import focused_knowledge
        for pair in data['feasible_measured_pairs']:
            pair['knowledge_evidence'] = kg_pair_evidence(
                data['knowledge'], p, pair['reference_variant'], pair['test_variant'])
        # Candidate IDs stay identical at a fixed history; ON adds evidence only.
        data['kg_summary'] = focused_knowledge(data['knowledge'])
    for idx, pair in enumerate(data['feasible_measured_pairs']):
        pair['pair_id'] = idx
    assert data['feasible_measured_pairs'], 'No eligible measured-reference pairs available'


    base_prompt = (PROMPT + '\n【合法 matched_comparison 选项】\n'
                   + json.dumps(data['feasible_measured_pairs'], ensure_ascii=False)
                   + '\n每条假设只输出 matched_comparison: {"pair_id": 合法编号}。'
                     '每条假设的 anchor_variant 必须是 hypothesis_anchors 指定的实测亲本，'
                      '所选 pair_id 的 reference_variant 必须与该亲本相同。')
    if 'knowledge' in data:
        base_prompt += (
            '\n【Knowledge ON：GB1-KG】\n'
            '根据 kg_summary 和 feasible_measured_pairs 中的 knowledge_evidence，'
            '提供可追溯的实验背景/理化属性/保守与激进替换解释。'
            '每条假设输出非空 knowledge_rationale 和 knowledge_falsification。'
            '解释中引用具体 R1-R7 规则或实际配对。不能把跨背景改善直接视为本背景的 Fitness。'
            '可基于 kg_combination_preview 提出双位点组合假设，由 Designer 另行生成，'
            '但 matched_comparison 仍是已测亲本的单点配对。'
            '上位性值不是候选的预测 Fitness。'
        )
    prompt = base_prompt
    rejected = []
    for attempt in range(3):
        result = generate_json(prompt, data)
        try:
            for hypothesis in result['hypotheses']:
                pair_id = int(hypothesis['matched_comparison']['pair_id'])
                assert 0 <= pair_id < len(data['feasible_measured_pairs']), (
                    f"{hypothesis['id']}: invalid pair_id {pair_id}"
                )
                canonical_pair = data['feasible_measured_pairs'][pair_id]
                hypothesis['matched_comparison'] = {
                    key: canonical_pair[key]
                    for key in ('test_variant', 'reference_variant',
                                'changed_position', 'reference_role')
                }
                if 'knowledge' in data:
                    rationale = required_knowledge_text(hypothesis, 'knowledge_rationale')
                    falsification = required_knowledge_text(hypothesis, 'knowledge_falsification')
                    hypothesis['knowledge_rationale'] = rationale
                    hypothesis['knowledge_falsification'] = falsification
                    evidence = kg_pair_evidence(
                        data['knowledge'], p, canonical_pair['reference_variant'],
                        canonical_pair['test_variant'])
                    hypothesis['knowledge_design'] = {
                        'selected_pair_evidence': evidence,
                        'selection_basis': canonical_pair.get('selection_basis', 'm0_local'),
                        'llm_knowledge_rationale': rationale,
                        'llm_knowledge_falsification': falsification,
                    }
                    from kg_agent_knowledge import verified_evidence_text
                    hypothesis['knowledge_design']['verified_evidence_text'] = verified_evidence_text(evidence)
                    hypothesis['knowledge_design']['evidence_ids'] = [
                        row['evidence_id'] for row in
                        evidence['other_background_examples'] + evidence['epistasis_examples']]
                    assert evidence is not None
                    assert hypothesis['knowledge_design']['llm_knowledge_rationale']
                    assert hypothesis['knowledge_design']['llm_knowledge_falsification']
            validate_round_plan(result, data, measured, available)
            for item in result['previous_hypotheses_review']:
                item['verified_comparison'] = data['previous_review_facts'][item['previous_id']]
            validate_hypothesis_claims(result['hypotheses'], observed)
            validate_text_claims(result, observed, previous_hypotheses=data['previous_hypotheses'])
            if data['previous_scientific_feedback'] is not None:
                earlier = {
                    row['hypothesis_id']: row
                    for row in data['previous_scientific_feedback']['hypotheses']
                }
                for item in result['previous_hypotheses_review']:
                    item['critic_pre_experiment_assessment'] = earlier[item['previous_id']][
                        'critic_pre_experiment_assessment'
                    ]
            break
        except (AssertionError, KeyError, ValueError) as exc:
            rejected.append(str(exc))
            if attempt == 2:
                (p.folder / 'invalid_hypotheses.json').write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
                )
                raise ValueError(f'Hypothesis plan still invalid after 3 attempts: {exc}') from exc
            print(f'Hypothesis plan rejected: {exc}; regenerating...', flush=True)
            prompt = (base_prompt + '\n先前输出的错误，全部必须避免：\n' + '\n'.join(rejected) +
                '\n按完整 JSON 示例重新返回三条假设，保留所有必填字段。'
                'Knowledge ON 时每条假设顶层都须包含非空字符串 knowledge_rationale 和 knowledge_falsification；'
                '不要将它们放到 summary 或 knowledge_design 中。'
                '上一轮复盘须使用 previous_review_facts 的实际配对Δ；KG跨背景中位Δ与当前实测Δ分开说明。'
                '重新选择合法 pair_id；不要自己填写 test_variant 等序列。'
                'anchor_variant 必须等于各自 hypothesis_anchors 指定的 reference_variant。')

    p.hypotheses.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Hypothesis Generator complete: {p.hypotheses}')


if __name__ == '__main__':
    main()
