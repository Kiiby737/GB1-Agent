import argparse
import os
import subprocess
import sys

from agent_paths import AGENT_DIR, AGENT_INITIAL_FILE, KNOWLEDGE_BRANCH_START, round_paths
from search_strategy import initialize_search, normalize_start, search_state

STEPS = [
    '1.data_analyst.py',
    '2.hypothesis_generator.py',
    '3.mutation_designer.py',
    '4.fitness_evaluator.py',
    '5.scientific_critic.py',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, default=KNOWLEDGE_BRANCH_START)
    parser.add_argument('--knowledge', choices=['on', 'off'], default='off')
    parser.add_argument('--start-step', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--start-variant', '--start-sequence', type=normalize_start, default=None)
    parser.add_argument('--kg-validation-slots', type=int, choices=(0,), default=0,
                        help='0: common single/two-site allocation for ON and OFF; '
                             'ON-only slots are disabled')
    args = parser.parse_args()
    if args.kg_validation_slots and args.knowledge != 'on':
        parser.error('Reserved KG slots require --knowledge on')
    env = os.environ.copy()
    env['AGENT_KNOWLEDGE_MODE'] = args.knowledge
    env['GB1_KG_VALIDATION_SLOTS'] = str(args.kg_validation_slots)
    os.environ['AGENT_KNOWLEDGE_MODE'] = args.knowledge
    os.environ['GB1_KG_VALIDATION_SLOTS'] = str(args.kg_validation_slots)
    assert args.round >= 1, 'Round numbers start at 1'
    if not AGENT_INITIAL_FILE.exists():
        if args.round == 1 and args.start_step == 1:
            print('Creating shared independent Agent initial 20 experiments...', flush=True)
            command = [sys.executable, str(AGENT_DIR / '0.initialize_agent_experiments.py')]
            if args.start_variant is not None:
                command += ['--start-variant', args.start_variant]
            subprocess.run(command,
                           check=True, cwd=AGENT_DIR, env=env)
        else:
            raise FileNotFoundError(f'Missing Agent initial 20: {AGENT_INITIAL_FILE}')
    initialize_search(args.start_variant)
    p = round_paths(args.round)
    if p.experiments.exists() or (args.start_step == 1 and p.analysis.exists()):
        raise FileExistsError(f'Round already started/completed: {p.folder}; resume an unfinished step or use a new output directory')
    p.folder.mkdir(parents=True, exist_ok=True)
    state = search_state(p)
    print(f"Seed={state['start_variant']} | active parents={state['parent_variants']}", flush=True)
    print(f'=== Round {args.round} | M0-AL | knowledge={args.knowledge} | {p.folder} ===', flush=True)
    for i in range(args.start_step - 1, len(STEPS)):
        script = AGENT_DIR / STEPS[i]
        print(f'\n[{i + 1}/5] {script.name}', flush=True)
        subprocess.run([sys.executable, str(script), '--round', str(args.round)],
                       check=True, cwd=AGENT_DIR, env=env)
    print(f'\nFinished Round {args.round}. Recommendations: {p.topk}', flush=True)


if __name__ == '__main__':
    main()
