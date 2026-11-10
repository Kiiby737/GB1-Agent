
import json

import pandas as pd

from agent_paths import existing_round_folder, get_round_arg, knowledge_enabled, read_observed
from search_strategy import search_state, hamming

WT = 'VDGV'
POSITIONS = [39, 40, 41, 54]
TOP_K = 20


def mutations(variant):
    return ';'.join(f'{wt}{pos}{aa}' for wt, pos, aa in zip(WT, POSITIONS, variant) if aa != wt)


def site_stats(df, score):
    result = []
    for i, pos in enumerate(POSITIONS):
        stats = (df.assign(aa=df['variant'].str[i]).groupby('aa')[score]
                 .agg(['count', 'mean', 'max']).reset_index())
        for row in stats.itertuples(index=False):
            result.append({
                'mutation': f'{WT[i]}{pos}{row.aa}' if row.aa != WT[i] else f'{pos}_WT',
                'count': int(row.count),
                f'mean_{score}': round(float(row.mean), 4),
                f'max_{score}': round(float(row.max), 4),
            })
    return result


def top_variants(df, score):
    top = df.nlargest(TOP_K, score).copy()
    top['mutations'] = top['variant'].map(mutations)
    columns = ['variant', score, 'mutations']
    if score == 'predicted_fitness':
        columns.insert(2, 'uncertainty')
    return top[columns].round(4).to_dict('records')


def top100_vs_pool(predicted):
    top = predicted.nlargest(100, 'predicted_fitness')
    result = []
    for i, pos in enumerate(POSITIONS):
        top_counts = top['variant'].str[i].value_counts()
        pool_counts = predicted['variant'].str[i].value_counts()
        for aa, count in top_counts.items():
            top_fraction = count / len(top)
            pool_fraction = pool_counts[aa] / len(predicted)
            result.append({
                'mutation': f'{WT[i]}{pos}{aa}' if aa != WT[i] else f'{pos}_WT',
                'top_count': int(count),
                'top_fraction': round(float(top_fraction), 4),
                'pool_count': int(pool_counts[aa]),
                'pool_fraction': round(float(pool_fraction), 4),
                'top_vs_pool_ratio': round(float(top_fraction / pool_fraction), 4),
            })
    return result



def previous_scientific_feedback(prev_folder, previous, previous_hypotheses, observed):
    """Contrast PRE-experiment Critic comments with already observed Oracle labels."""
    review = json.loads((prev_folder / 'scientific_review.json').read_text(encoding='utf-8'))
    critic_by_h = {r['hypothesis_id']: r for r in review['hypothesis_assessments']}
    critic_by_variant = {r['variant']: r for r in review['reviews']}
    measured_fitness = dict(zip(observed['variant'], observed['fitness']))

    def one_site_delta(test, reference):
        if (test in measured_fitness and reference in measured_fitness
                and sum(a != b for a, b in zip(test, reference)) == 1):
            return round(float(measured_fitness[test] - measured_fitness[reference]), 4)
        return None

    selected = []
    for row in previous.to_dict('records'):
        v = row['variant']
        before = critic_by_variant.get(v, {})
        ref = row.get('effective_reference_variant', '')
        if not isinstance(ref, str) or not ref.strip():
            ref = row.get('reference_variant', '')
        ref = ref.strip().upper() if isinstance(ref, str) else ''
        delta = one_site_delta(v, ref) if ref else None
        selected.append({
            'variant': v,
            'hypothesis_id': row['hypothesis_id'],
            'purpose': row['purpose'],
            'predicted_fitness': round(float(row['predicted_fitness']), 4),
            'actual_fitness': round(float(row['fitness']), 4),
            'prediction_error': round(float(row['prediction_error']), 4),
            'critic_alignment_before_experiment': before.get('hypothesis_alignment', ''),
            'critic_reason_before_experiment': before.get('scientific_reason', ''),
            'critic_limitations_before_experiment': before.get('limitations', ''),
            'verified_reference': ref if delta is not None else None,
            'observed_delta_vs_reference': delta,
        })

    hypotheses_feedback = []
    for h in previous_hypotheses:
        hid = h['id']
        earlier = critic_by_h.get(hid, {})
        pair = h['matched_comparison']
        test = pair['test_variant']
        reference = pair['reference_variant']
        delta = one_site_delta(test, reference)
        tests = [r for r in selected if r['hypothesis_id'] == hid and r['purpose'] == 'test']
        # Prefer the mandatory comparison, then the most informative successes/failures.
        ranked = sorted(tests, key=lambda r: abs(r['prediction_error']), reverse=True)
        example_rows = [r for r in ranked if r['variant'] == test]
        example_rows += [r for r in ranked if r['variant'] != test][:2]
        hypotheses_feedback.append({
            'hypothesis_id': hid,
            'critic_pre_experiment_assessment': earlier.get('assessment', ''),
            'critic_pre_experiment_reason': earlier.get('reason', ''),
            'mandatory_matched_comparison': {
                'test_variant': test,
                'reference_variant': reference,
                'test_fitness': round(float(measured_fitness[test]), 4) if test in measured_fitness else None,
                'reference_fitness': round(float(measured_fitness[reference]), 4) if reference in measured_fitness else None,
                'observed_single_site_delta': delta,
                'note': ('Observed single-site contrast, not proof of general mutation benefit'
                         if delta is not None else 'No fully observed single-site contrast'),
            },
            'selected_test_examples': example_rows,
        })
    return {
        'source': str(prev_folder / 'scientific_review.json'),
        'evaluation_timing': 'Critic assessments were made BEFORE the last-round Oracle labels',
        'hypotheses': hypotheses_feedback,
        'all_selected_candidate_outcomes': selected,
    }

def main():
    p = get_round_arg()
    observed = read_observed(p)
    state = search_state(p)
    predicted = pd.read_csv(p.predictions, usecols=['variant', 'predicted_fitness', 'uncertainty'])
    predicted['variant'] = predicted['variant'].astype(str).str.upper().str.strip()
    assert predicted['variant'].is_unique, 'Duplicate variants in current predictions'
    test = set(pd.read_csv(p.test, usecols=['variant'])['variant'].str.upper())
    assert not set(observed['variant']) & test, 'Held-out test variants leaked into measured data'
    excluded = set(observed['variant']) | test
    from common_search import candidate_pool
    predicted = candidate_pool(predicted, state['parent_variants'], set(observed.variant), test)
    assert len(predicted) > 0, 'No eligible candidate predictions remain'

    last_round = None
    previous_hypotheses = []
    if p.round_num > 1:
        prev_folder = existing_round_folder(p.round_num - 1)
        previous = pd.read_csv(prev_folder / 'experiment_results.csv')
        previous = previous.rename(columns={'hypothesis_id_x': 'hypothesis_id'})
        last_round = {
            'round': p.round_num - 1,
            'mean_fitness': round(float(previous['fitness'].mean()), 4),
            'max_fitness': round(float(previous['fitness'].max()), 4),
            'results': previous[[
                'variant', 'hypothesis_id', 'purpose',
                'predicted_fitness', 'fitness', 'prediction_error'
            ]].round(4).to_dict('records'),
        }
        previous_hypotheses = json.loads(
            (prev_folder / 'hypotheses.json').read_text(encoding='utf-8')
        )['hypotheses']
        feedback = previous_scientific_feedback(
            prev_folder, previous, previous_hypotheses, observed
        )
        (p.folder / 'previous_scientific_feedback.json').parent.mkdir(parents=True, exist_ok=True)
        (p.folder / 'previous_scientific_feedback.json').write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    report = {
        'last_round': last_round,
        'previous_hypotheses': previous_hypotheses,
        'previous_scientific_feedback': (
            {'source': feedback['source'],
             'evaluation_timing': feedback['evaluation_timing'],
             'hypotheses': feedback['hypotheses']}
            if p.round_num > 1 else None
        ),
        'round': p.round_num,
        'model': p.model_name,
        'search_state': state,
        'history_sources': {
            'm0_training_knowledge': len(pd.read_csv(p.m0_history, usecols=['variant'])),
            'agent_initial_knowledge': len(pd.read_csv(p.initial)),
            'seed_variant': state['start_variant'],
            'current_search_parent': state['parent_variant'],
            'active_parents': state['parent_variants'],
            'earlier_rounds': p.round_num - 1,
            'new_experiments_in_current_round': 0,
        },
        'observed': {
            'count': len(observed),
            'mean_fitness': round(float(observed['fitness'].mean()), 4),
            'max_fitness': round(float(observed['fitness'].max()), 4),
            'top_variants': top_variants(observed, 'fitness'),
            'site_stats': site_stats(observed, 'fitness'),
        },
        'predicted': {
            'count': len(predicted),
            'mean_predicted_fitness': round(float(predicted['predicted_fitness'].mean()), 4),
            'max_predicted_fitness': round(float(predicted['predicted_fitness'].max()), 4),
            'top_variants': top_variants(predicted, 'predicted_fitness'),
            'top100_site_stats': site_stats(predicted.nlargest(100, 'predicted_fitness'), 'predicted_fitness'),
            'top100_vs_pool': top100_vs_pool(predicted),
        },
    }
    report['common_two_site_preview'] = predicted.loc[predicted.mutation_distance_to_anchor.eq(2)].head(24).to_dict('records')
    if knowledge_enabled():
        from kg_agent_knowledge import build_knowledge
        report['knowledge'] = build_knowledge(p)
        from kg_agent_knowledge import kg_combination_proposals
        pool_scores = pd.read_csv(p.predictions, usecols=['variant', 'predicted_fitness'])
        full_scores = dict(zip(pool_scores['variant'], pool_scores['predicted_fitness']))
        allowed = set(full_scores) - excluded
        combo_preview = kg_combination_proposals(
            state['parent_variants'], allowed, full_scores, p, per_anchor=12)
        report['kg_combination_candidate_count'] = len(combo_preview)
        report['kg_combination_preview'] = combo_preview[:12]
        (p.folder / 'knowledge_context.json').parent.mkdir(parents=True, exist_ok=True)
        (p.folder / 'knowledge_context.json').write_text(
            json.dumps(report['knowledge'], ensure_ascii=False, indent=2), encoding='utf-8'
        )

    p.folder.mkdir(parents=True, exist_ok=True)
    p.analysis.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Round {p.round_num} ({p.model_name}): {len(observed)} measured, {len(predicted)} unmeasured')
    print(f'Data Analyst complete: {p.analysis}')


if __name__ == '__main__':
    main()
