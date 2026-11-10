import argparse
import json
import os
import subprocess
import sys

import pandas as pd

from agent_paths import AGENT_DIR, AGENT_INITIAL_FILE, existing_round_folder, rounds_dir
from search_strategy import PROTOCOL, initialize_search, normalize_start, load_start


def summarize(last_round):
    start = load_start()
    initial = pd.read_csv(AGENT_INITIAL_FILE, usecols=['variant', 'fitness'])
    initial_best = max(float(initial['fitness'].max()), float(start['fitness']))
    new_best_so_far = float('-inf')
    previous_best, previous_mean = None, None
    rows = []
    for r in range(1, last_round + 1):
        data = pd.read_csv(existing_round_folder(r) / 'experiment_results.csv')
        state = json.loads((existing_round_folder(r) / 'search_state.json').read_text(encoding='utf-8'))
        best = float(data['fitness'].max())
        mean = float(data['fitness'].mean())
        new_best_so_far = max(new_best_so_far, best)
        rows.append({
            'round': r,
            'model': 'M0-AL',
            'knowledge_mode': ('knowledge' if os.environ.get('AGENT_KNOWLEDGE_MODE') == 'on'
                               else 'no_knowledge'),
            'n_measured': len(data),
            'round_best_fitness': best,
            'round_mean_fitness': mean,
            'round_median_fitness': float(data['fitness'].median()),
            'agent_best_so_far': max(initial_best, new_best_so_far),
            'new_experiment_best_so_far': new_best_so_far,
            'initial_agent_best_fitness': initial_best,
            'search_protocol': PROTOCOL,
            'reserved_kg_slots': int(state.get('kg_validation_slots', 0)),
            'active_parent_count': len(state['parent_variants']),
            'active_parents': ';'.join(state['parent_variants']),
            'selected_two_site': int(data['mutation_distance_to_anchor'].eq(2).sum()),
            'selected_kg_evidence': int(data.get('kg_evidence_id', pd.Series('',index=data.index)).fillna('').ne('').sum()),
            'selected_multi_parent_variants': int(data['anchor_variant'].nunique()),
            'start_variant': start['variant'],
            'seed_oracle_queries': start['seed_oracle_queries'],
            'cumulative_new_experiments': r * 20,
            'cumulative_total_experiments': len(initial) + start['seed_oracle_queries'] + r * 20,
            'delta_best_vs_previous_round': (best - previous_best if previous_best is not None else float('nan')),
            'delta_mean_vs_previous_round': (mean - previous_mean if previous_mean is not None else float('nan')),
            'count_above_wt': int((data['fitness'] > 1).sum()),
            'n_test': int((data['purpose'] == 'test').sum()),
            'n_control': int((data['purpose'] == 'control').sum()),
        })
        previous_best, previous_mean = best, mean
    output = rounds_dir() / 'evolution_summary.csv'
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f'Agent evolution summary: {output}')
    print(frame[['round', 'knowledge_mode', 'round_best_fitness', 'round_mean_fitness',
                 'new_experiment_best_so_far']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--knowledge', choices=['on', 'off'], default='off')
    parser.add_argument('--start-variant', '--start-sequence', type=normalize_start, default=None)
    args = parser.parse_args()
    # Preserve the policy already recorded by run_agent, including standalone use.
    env = os.environ.copy()
    env['AGENT_KNOWLEDGE_MODE'] = args.knowledge
    os.environ['AGENT_KNOWLEDGE_MODE'] = args.knowledge
    state_file = existing_round_folder(args.round) / 'search_state.json'
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding='utf-8'))
        env['GB1_KG_VALIDATION_SLOTS'] = str(state.get('kg_validation_slots', 0))
        os.environ['GB1_KG_VALIDATION_SLOTS'] = env['GB1_KG_VALIDATION_SLOTS']
    initialize_search(args.start_variant)
    subprocess.run([sys.executable, str(AGENT_DIR / '6.virtual_experiment.py'),
                    '--round', str(args.round)], check=True, cwd=AGENT_DIR, env=env)
    summarize(args.round)
    print(f'Next: python run_agent.py --round {args.round + 1} --knowledge {args.knowledge}')


if __name__ == '__main__':
    main()
