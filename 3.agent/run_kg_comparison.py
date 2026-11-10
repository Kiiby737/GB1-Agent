"""Compare KG decisions on identical frozen histories, separately from evolution.

All arms finish and lock their selections before this module queries the Oracle.
Copied prefix rounds are inherited observations, never new experiments. A trial
does not feed either arm's new labels back into the source trajectory.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import agent_paths as ap
from search_strategy import PROTOCOL, BATCH_SIZE, hamming
from sequence_context import file_sha256, validate_prediction_pool, training_template, sequence_for_variant

PREFIX_FILES = ('search_state.json', 'experiment_results.csv', 'hypotheses.json', 'scientific_review.json')


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def arm_folder(trial, arm, number):
    mode = 'no_knowledge' if arm == 'off' else 'knowledge'
    return Path(trial) / arm / mode / f'round_{number:03d}'



def prepare_trial(source, history_branch, number, trial, include_reserved=False):
    if include_reserved:
        raise ValueError('ON-only reserved controls are disabled in the common-search protocol')
    source, trial = Path(source).resolve(), Path(trial).resolve()
    if trial.exists():
        raise FileExistsError(f'Refusing to overwrite an existing comparison trial: {trial}')
    validate_prediction_pool(ap.POOL_PREDICTIONS, ap.INITIAL_FILE, ap.MODEL_FILE)
    start = json.loads((source / 'search_start.json').read_text(encoding='utf-8'))
    inputs = {'m0': ap.INITIAL_FILE, 'initial20': source / 'agent_initial_labeled.csv',
              'heldout': ap.TEST_FILE, 'prediction_pool': ap.POOL_PREDICTIONS, 'oracle': ap.ORACLE_FILE}
    fingerprints = {key: file_sha256(path) for key, path in inputs.items()}
    if start.get('protocol') != PROTOCOL or start.get('input_sha256') != fingerprints:
        raise ValueError('Source run has an old protocol or changed inputs; rebuild it first')
    prefix = []
    for previous in range(1, number):
        folder = source / history_branch / f'round_{previous:03d}'
        state = json.loads((folder / 'search_state.json').read_text(encoding='utf-8'))
        if state.get('protocol') != PROTOCOL or state.get('kg_validation_slots', 0) != 0:
            raise ValueError('Use a source history from the main zero-reserved-slot protocol')
        for name in PREFIX_FILES:
            path = folder / name
            if not path.exists():
                raise FileNotFoundError(path)
            prefix.append((previous, name, path))
    arms = ['off', 'on']
    # Freeze the KG files too, so an edit cannot change the ON arms mid-trial.
    kg_files = [ap.KG_DIR / 'kg_graph.json', ap.KG_DIR / 'kg_measured_pairs.csv']
    # Extra static exports are optional; per-round evidence is rebuilt from measured history.
    kg_files += [ap.KG_DIR / name for name in (
        'kg_directed_pairs.csv', 'kg_mutation_combinations.csv', 'kg_epistasis.csv')
        if (ap.KG_DIR / name).exists()]
    for path in kg_files:
        if not path.exists():
            raise FileNotFoundError(path)
    trial.mkdir(parents=True)
    for path in kg_files:
        target = trial / 'kg_snapshot' / path.relative_to(ap.KG_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for arm in arms:
        target = trial / arm
        target.mkdir()
        shutil.copy2(source / 'agent_initial_labeled.csv', target / 'agent_initial_labeled.csv')
        shutil.copy2(source / 'search_start.json', target / 'search_start.json')
        if (source / 'agent_initial_metadata.json').exists():
            shutil.copy2(source / 'agent_initial_metadata.json', target / 'agent_initial_metadata.json')
        mode = 'no_knowledge' if arm == 'off' else 'knowledge'
        for previous, name, path in prefix:
            folder = target / mode / f'round_{previous:03d}'
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, folder / name)
    context = {
        'protocol': PROTOCOL, 'experiment_kind': 'paired_same_history',
        'source_results': str(source), 'history_branch': history_branch, 'round': number,
        'arms': arms, 'input_sha256': fingerprints,
        'prefix_sha256': {f'round_{n:03d}/{name}': file_sha256(path) for n, name, path in prefix},
        'kg_sha256': {str(p.relative_to(ap.KG_DIR)): file_sha256(p) for p in kg_files},
        'model_sha256': file_sha256(ap.MODEL_FILE),
        'llm_model': os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash'),
        'note': 'Inherited prefix labels are fixed; no current-round Oracle labels are supplied to any arm.',
    }
    save_json(trial / 'context.json', context)
    return context


def compare_trial(trial):
    trial = Path(trial)
    context = json.loads((trial / 'context.json').read_text(encoding='utf-8'))
    number = context['round']
    current = {'m0': ap.INITIAL_FILE, 'initial20': Path(context['source_results']) / 'agent_initial_labeled.csv',
               'heldout': ap.TEST_FILE, 'prediction_pool': ap.POOL_PREDICTIONS, 'oracle': ap.ORACLE_FILE}
    if any(file_sha256(path) != context['input_sha256'][key] for key, path in current.items()):
        raise ValueError('Comparison inputs changed after freezing the history')
    if file_sha256(ap.MODEL_FILE) != context['model_sha256']:
        raise ValueError('M0 checkpoint changed during the comparison')
    for name, digest in context['kg_sha256'].items():
        if file_sha256(trial / 'kg_snapshot' / name) != digest:
            raise ValueError('Frozen KG changed during comparison')
    scores = pd.read_csv(ap.POOL_PREDICTIONS).set_index('variant')['predicted_fitness']
    selections, designed, hypotheses, states, locks = {}, {}, {}, {}, {}
    common = None
    observed = set(pd.read_csv(ap.INITIAL_FILE)['variant'])
    observed.update(pd.read_csv(current['initial20'])['variant'])
    start = json.loads((Path(context['source_results']) / 'search_start.json').read_text(encoding='utf-8'))
    observed.add(start['variant'])
    heldout = set(pd.read_csv(ap.TEST_FILE)['variant'])
    for arm in context['arms']:
        folder = arm_folder(trial, arm, number)
        mode = folder.parent.name
        for name, digest in context['prefix_sha256'].items():
            if file_sha256(trial / arm / mode / name) != digest:
                raise ValueError('An arm changed its frozen historical prefix')
        for previous in range(1, number):
            observed.update(pd.read_csv(trial / arm / mode / f'round_{previous:03d}/experiment_results.csv')['variant'])
        state = json.loads((folder / 'search_state.json').read_text(encoding='utf-8'))
        shared = {k: state[k] for k in ('protocol', 'start_variant', 'parent_variant', 'parent_fitness',
                                       'parent_variants',
                                       'eligible_neighbor_count', 'admissible_pool_sha256', 'two_site_target', 'previous_round_sha256')}
        if common is not None and shared != common:
            raise ValueError('Arms do not share the same parent/history/candidate neighborhood')
        common = shared
        expected_slots = 0
        if state.get('kg_validation_slots') != expected_slots:
            raise ValueError('An arm used the wrong selection policy')
        selected = pd.read_csv(folder / 'topk_recommendations.csv')
        if len(selected) != BATCH_SIZE or not selected['variant'].is_unique:
            raise ValueError('Every arm must lock 20 unique selections')
        if set(selected['variant']) & (observed | heldout):
            raise ValueError('An arm selected measured or held-out variants')
        from common_search import domain, domain_hash
        assigned = domain(state['parent_variants'],set(scores.index)-observed-heldout)
        if domain_hash(assigned)!=state['admissible_pool_sha256']:
            raise ValueError('Common search-domain fingerprint changed')
        if any(assigned.get(r.variant)!=(r.anchor_variant,int(r.mutation_distance_to_anchor))
               for r in selected.itertuples(index=False)):
            raise ValueError('A selected variant left the shared radius-two domain')
        if int(selected.mutation_distance_to_anchor.eq(2).sum())!=state['two_site_target']:
            raise ValueError('A comparison arm changed the common mutation-order allocation')
        if arm=='off' and selected.get('kg_evidence_id',pd.Series('',index=selected.index)).fillna('').ne('').any():
            raise ValueError('OFF arm carries KG evidence')
        expected = selected['variant'].map(scores)
        if not np.isfinite(expected).all() or not np.allclose(expected, selected['predicted_fitness'], atol=1e-7, rtol=0):
            raise ValueError('An arm modified the frozen M0 predictions')
        selections[arm], states[arm] = selected, state
        designed[arm] = pd.read_csv(folder / 'designed_candidates.csv')
        hypotheses[arm] = json.loads((folder / 'hypotheses.json').read_text(encoding='utf-8'))
        locks[arm] = file_sha256(folder / 'topk_recommendations.csv')
    # Selection is immutable and recorded BEFORE any current-round label read.
    save_json(trial / 'selection_lock.json', locks)
    union = sorted(set().union(*(set(frame['variant']) for frame in selections.values())))
    oracle = pd.read_csv(ap.ORACLE_FILE, usecols=['AACombo', 'Fitness']).rename(
        columns={'AACombo': 'variant', 'Fitness': 'fitness'})
    if not oracle['variant'].is_unique:
        raise ValueError('Oracle contains duplicate variants')
    truth = pd.DataFrame({'variant': union}).merge(oracle, on='variant', how='left', validate='one_to_one')
    if not np.isfinite(truth['fitness']).all():
        raise ValueError('Missing/non-finite Oracle labels')
    template = training_template(ap.INITIAL_FILE)
    rows, decision = [], {'common_context': common, 'comparisons': {}, 'arms': {}}
    for arm, selected in selections.items():
        if file_sha256(arm_folder(trial, arm, number) / 'topk_recommendations.csv') != locks[arm]:
            raise ValueError('Selections changed after Oracle access')
        result = selected.merge(truth, on='variant', how='left', validate='one_to_one')
        result['mutant_seq'] = result['variant'].map(lambda v: sequence_for_variant(v, template))
        result['prediction_error'] = result['predicted_fitness'] - result['fitness']
        result.to_csv(arm_folder(trial, arm, number) / 'experiment_results.csv', index=False)
        rows.append({'round': number, 'arm': arm, 'parent': common['parent_variant'],
                     'parent_fitness': common['parent_fitness'], 'mean_fitness': float(result['fitness'].mean()),
                     'best_fitness': float(result['fitness'].max()), 'median_fitness': float(result['fitness'].median()),
                     'fraction_above_parent': float(result['fitness'].gt(common['parent_fitness']).mean()),
                     'mean_predicted_fitness': float(result['predicted_fitness'].mean()),
                     'selected_count': len(result), 'reserved_kg_slots': 0})
        decision['arms'][arm] = {
            'candidate_count': len(designed[arm]),
            'candidate_methods': designed[arm]['design_method'].value_counts().to_dict(),
            'selected_methods': selected['design_method'].value_counts().to_dict(),
            'selected_two_site': int(selected['mutation_distance_to_anchor'].eq(2).sum()),
            'hypotheses': hypotheses[arm], 'selected_variants': selected['variant'].tolist(),
        }
    for left, right in [('off', 'on')]:
        a, b = set(selections[left]['variant']), set(selections[right]['variant'])
        candidates_a, candidates_b = set(designed[left]['variant']), set(designed[right]['variant'])
        pairs_a = {(h['matched_comparison']['reference_variant'], h['matched_comparison']['test_variant'])
                   for h in hypotheses[left]['hypotheses']}
        pairs_b = {(h['matched_comparison']['reference_variant'], h['matched_comparison']['test_variant'])
                   for h in hypotheses[right]['hypotheses']}
        decision['comparisons'][f'{right}_vs_{left}'] = {
            'top20_overlap_count': len(a & b), 'changed_selection_count': BATCH_SIZE - len(a & b),
            'top20_jaccard': len(a & b) / len(a | b),
            'candidate_jaccard': len(candidates_a & candidates_b) / len(candidates_a | candidates_b),
            'hypothesis_pair_overlap_count': len(pairs_a & pairs_b),
            'frozen_m0_scores_verified_against_common_pool': True,
            'selected_only_left': sorted(a - b), 'selected_only_right': sorted(b - a),
        }
    metrics = pd.DataFrame(rows)
    off = metrics.set_index('arm').loc['off']
    metrics['mean_fitness_difference_vs_off'] = metrics['mean_fitness'] - off['mean_fitness']
    metrics['best_fitness_difference_vs_off'] = metrics['best_fitness'] - off['best_fitness']
    metrics.to_csv(trial / 'paired_fitness_metrics.csv', index=False)
    truth.to_csv(trial / 'queried_union.csv', index=False)
    decision['new_unique_oracle_labels'] = len(union)
    decision['interpretation'] = ('Conditional same-history common-radius-two and common-allocation KG-input comparison. One realization is not proof of a systematic '
                                  'KG benefit; use repeated trials. Shadow labels never update the source trajectory.')
    save_json(trial / 'decision_comparison.json', decision)
    return metrics


def run_trial(source, history_branch, number, trial, include_reserved=False, runner=None):
    context = prepare_trial(source, history_branch, number, trial, include_reserved)
    runner = runner or subprocess.run
    for arm in context['arms']:
        mode = 'off' if arm == 'off' else 'on'
        slots = 0
        env = dict(os.environ, GB1_AGENT_RESULT_DIR=str(Path(trial).resolve() / arm),
                   GB1_KG_DIR=str(Path(trial).resolve() / 'kg_snapshot'),
                   AGENT_KNOWLEDGE_MODE=mode, GB1_KG_VALIDATION_SLOTS=str(slots))
        runner([sys.executable, str(ap.AGENT_DIR / 'run_agent.py'), '--round', str(number),
                '--knowledge', mode, '--kg-validation-slots', str(slots)],
               check=True, cwd=ap.AGENT_DIR, env=env)
    return compare_trial(trial)


def aggregate_trials(output):
    tables = []
    for path in sorted(Path(output).glob('round_*/repeat_*/paired_fitness_metrics.csv')):
        table = pd.read_csv(path)
        table['repeat'] = int(path.parent.name.split('_')[-1])
        tables.append(table)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(Path(output) / 'paired_fitness_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-results', type=Path, default=ap.AGENT_RESULT_DIR)
    parser.add_argument('--history-branch', choices=('no_knowledge', 'knowledge'), default='no_knowledge')
    parser.add_argument('--start-round', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if min(args.start_round, args.rounds, args.repeats) < 1:
        parser.error('Round numbers, counts and repeats must be positive')
    output = args.output or args.source_results / 'kg_same_history_comparison'
    output.mkdir(parents=True, exist_ok=True)
    config = {'protocol': PROTOCOL, 'source_results': str(args.source_results.resolve()),
              'history_branch': args.history_branch, 'include_reserved_control': False,
              'llm_model': os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash')}
    config_path = output / 'comparison_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Comparison configuration changed; choose a new --output directory')
    else:
        if any(output.glob('round_*/repeat_*')):
            raise ValueError('Existing trials have no comparison configuration; choose a new --output')
        save_json(config_path, config)
    for number in range(args.start_round, args.start_round + args.rounds):
        for repeat in range(1, args.repeats + 1):
            trial = output / f'round_{number:03d}' / f'repeat_{repeat:03d}'
            run_trial(args.source_results, args.history_branch, number, trial, False)
            aggregate_trials(output)
            print(f'Completed same-history comparison: round={number}, repeat={repeat}, {trial}', flush=True)


if __name__ == '__main__':
    main()
