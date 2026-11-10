import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('GB1_V44_ROOT', AGENT_DIR.parent)).resolve()
_local_data = ROOT if (ROOT / '2.train/hybrid_30_40_30_r9/M0_AL_labeled.csv').exists() else ROOT.parent / 'ver4.3'
DATA_ROOT = Path(os.environ.get('GB1_DATA_ROOT', _local_data if _local_data.exists() else ROOT)).resolve()
KG_DIR = Path(os.environ.get('GB1_KG_DIR', ROOT / '6.knowledge'
              if (ROOT / '6.knowledge').exists() else DATA_ROOT / '6.knowledge')).resolve()
RESULT_DIR = DATA_ROOT / '4.result'
AGENT_RESULT_DIR = Path(os.environ.get('GB1_AGENT_RESULT_DIR', ROOT / '4.result_ver1')).resolve()
BASE_ROUNDS_DIR = AGENT_RESULT_DIR / 'no_knowledge'
KG_ROUNDS_DIR = AGENT_RESULT_DIR / 'knowledge'


INITIAL_FILE = DATA_ROOT / '2.train/hybrid_30_40_30_r9/M0_AL_labeled.csv'
AGENT_INITIAL_FILE = AGENT_RESULT_DIR / 'agent_initial_labeled.csv'
AGENT_INITIAL_METADATA = AGENT_RESULT_DIR / 'agent_initial_metadata.json'
AGENT_INITIAL_SIZE = 20
AGENT_INITIAL_SEED = 114514

MODEL_FILE = DATA_ROOT / '2.train/hybrid_30_40_30_r9/M0_AL_ensemble.pt'
TEST_FILE = RESULT_DIR / 'flip_test_variants.csv'
FLIP_FILE = DATA_ROOT / '1.data/sampled.csv'
ORACLE_FILE = DATA_ROOT / '1.data/gb1_2016_149361_seq.csv'
EMBEDDING_CACHE = RESULT_DIR / 'gb1_original_esm2_site4_float16.npy'
POOL_PREDICTIONS = RESULT_DIR / 'gb1_original_pool_predictions.csv'
KNOWLEDGE_BRANCH_START = 1


def knowledge_enabled():
    return os.environ.get('AGENT_KNOWLEDGE_MODE', 'off') == 'on'


def kg_validation_slots():
    value = int(os.environ.get('GB1_KG_VALIDATION_SLOTS', '0'))
    if value != 0:
        raise ValueError('Common-search protocol has no ON-only reserved slots; use 0')
    return value


def rounds_dir():
    return KG_ROUNDS_DIR if knowledge_enabled() else BASE_ROUNDS_DIR


def presentation_result_dir(project_root, explicit=None):
    import json
    from search_strategy import PROTOCOL
    result = Path(explicit or os.environ.get('GB1_AGENT_RESULT_DIR', Path(project_root) / '4.result')).resolve()
    start = json.loads((result / 'search_start.json').read_text(encoding='utf-8'))
    if start.get('protocol') != PROTOCOL:
        raise ValueError('Legacy result protocol; select the new experiment directory for presentation')
    return result


def existing_round_folder(round_num):
    return rounds_dir() / f'round_{round_num:03d}'


def round_paths(round_num):
    folder = existing_round_folder(round_num)
    return SimpleNamespace(
        round_num=round_num,
        knowledge_mode='on' if knowledge_enabled() else 'off',
        model_name='M0-AL',
        model_file=MODEL_FILE,
        initial=AGENT_INITIAL_FILE,
        m0_history=INITIAL_FILE,
        test=TEST_FILE,
        folder=folder,
        predictions=POOL_PREDICTIONS,
        analysis=folder / 'data_analysis.json',
        hypotheses=folder / 'hypotheses.json',
        designed=folder / 'designed_candidates.csv',
        scored=folder / 'scored_candidates.csv',
        review=folder / 'scientific_review.json',
        topk=folder / 'topk_recommendations.csv',
        experiments=folder / 'experiment_results.csv',
    )


def get_round_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, default=KNOWLEDGE_BRANCH_START)
    parser.add_argument('--knowledge', choices=['on', 'off'], default=None)
    parser.add_argument('--start-variant', '--start-sequence', default=None)
    args = parser.parse_args()
    if args.round < 1:
        parser.error('--round must be positive')
    if args.knowledge:
        os.environ['AGENT_KNOWLEDGE_MODE'] = args.knowledge
    from search_strategy import initialize_search
    initialize_search(args.start_variant)
    return round_paths(args.round)


def read_observed(paths):
    """读取 M0、共享初始20、起点基线和本分支严格早于本轮的实验标签。"""
    batches = [
        pd.read_csv(INITIAL_FILE, usecols=['variant', 'fitness']),
        pd.read_csv(AGENT_INITIAL_FILE, usecols=['variant', 'fitness']),
    ]
    for earlier in range(1, paths.round_num):
        batches.append(pd.read_csv(
            existing_round_folder(earlier) / 'experiment_results.csv',
            usecols=['variant', 'fitness']))
    observed = pd.concat(batches, ignore_index=True)
    observed['variant'] = observed['variant'].astype(str).str.upper().str.strip()
    from search_strategy import load_start
    start = load_start()
    if start['variant'] not in set(observed['variant']):
        observed = pd.concat([observed, pd.DataFrame([{
            'variant': start['variant'], 'fitness': start['fitness']
        }])], ignore_index=True)
    else:
        import math
        recorded = float(observed.loc[observed['variant'] == start['variant'], 'fitness'].iloc[0])
        if not math.isclose(recorded, start['fitness'], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Starting sequence label changed within this experiment')
    if not observed['variant'].is_unique:
        raise ValueError('重复历史测量：请检查 M0 / Agent初始20 / 起点 / 各轮实验')
    held_out = set(pd.read_csv(TEST_FILE, usecols=['variant'])['variant'].astype(str).str.upper().str.strip())
    if set(observed['variant']) & held_out:
        raise ValueError('独立测试变体不允许作为 Agent 历史实验')
    return observed
