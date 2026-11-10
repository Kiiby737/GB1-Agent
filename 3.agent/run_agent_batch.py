import argparse
import os
import subprocess
import sys

from agent_paths import AGENT_DIR, existing_round_folder, round_paths
from search_strategy import normalize_start


def main():
    parser = argparse.ArgumentParser(description='Run sequential GB1 Agent rounds')
    parser.add_argument('--start-round', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--knowledge', choices=['on', 'off'], default='on')
    parser.add_argument('--start-variant', '--start-sequence', type=normalize_start, default=None)
    parser.add_argument('--kg-validation-slots', type=int, choices=(0,), default=0,
                        help='Both modes use the shared target of 6 two-site tests; only value 0 is supported')
    args = parser.parse_args()
    if args.kg_validation_slots and args.knowledge != 'on':
        parser.error('Reserved KG slots require --knowledge on')

    assert args.start_round >= 1, 'Round numbers start from 1'
    assert args.rounds >= 1, '--rounds must be positive'

    env = os.environ.copy()
    env['AGENT_KNOWLEDGE_MODE'] = args.knowledge
    env['GB1_KG_VALIDATION_SLOTS'] = str(args.kg_validation_slots)
    os.environ['AGENT_KNOWLEDGE_MODE'] = args.knowledge
    os.environ['GB1_KG_VALIDATION_SLOTS'] = str(args.kg_validation_slots)

    if args.start_round > 1:
        previous = existing_round_folder(args.start_round - 1) / 'experiment_results.csv'
        if not previous.exists():
            raise FileNotFoundError(f'Previous round not finished: {previous}')

    last_round = args.start_round + args.rounds - 1
    start_args = ['--start-variant', args.start_variant] if args.start_variant is not None else []
    for round_num in range(args.start_round, last_round + 1):
        paths = round_paths(round_num)
        # Never silently overwrite an already completed or partially generated round.
        existing_files = [
            paths.analysis, paths.hypotheses, paths.designed,
            paths.scored, paths.review, paths.topk, paths.experiments,
        ]
        if any(file.exists() for file in existing_files):
            raise FileExistsError(
                f'Round {round_num} already has files in {paths.folder}; '
                'use the next unstarted round, or resume this round manually '
                'with run_agent.py --start-step N.'
            )

        print(f'\n=== Round {round_num}/{last_round} | knowledge={args.knowledge} ===', flush=True)
        subprocess.run(
            [sys.executable, str(AGENT_DIR / 'run_agent.py'),
             '--round', str(round_num), '--knowledge', args.knowledge,
             '--kg-validation-slots', str(args.kg_validation_slots)] + start_args,
            cwd=AGENT_DIR, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, str(AGENT_DIR / 'run_evolution_round.py'),
             '--round', str(round_num), '--knowledge', args.knowledge] + start_args,
            cwd=AGENT_DIR, env=env, check=True,
        )
        assert paths.experiments.exists(), f'Missing experiment results: {paths.experiments}'
        print(f'Completed Round {round_num}: {paths.experiments}', flush=True)

    print(f'\nFinished rounds {args.start_round}–{last_round} ({args.knowledge}).', flush=True)


if __name__ == '__main__':
    main()
