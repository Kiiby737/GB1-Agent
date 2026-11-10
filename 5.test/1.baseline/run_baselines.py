"""Local Random/M0 baselines using the same radius-two policy as both Agents."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'3.agent'))
import agent_paths as ap
from common_search import candidate_pool, select_batch
from search_strategy import initialize_search, load_start, search_state, PROTOCOL


def branch_path(method,seed):
    return ap.AGENT_RESULT_DIR/'baselines'/('m0_only' if method=='m0' else f'random_search_seed{seed}')


def observed_history(branch,number):
    start=load_start()
    frames=[pd.read_csv(ap.INITIAL_FILE)[['variant','fitness']],
            pd.read_csv(ap.AGENT_INITIAL_FILE)[['variant','fitness']]]
    for n in range(1,number):
        frames.append(pd.read_csv(branch/f'round_{n:03d}/experiment_results.csv')[['variant','fitness']])
    observed=pd.concat(frames,ignore_index=True)
    if start['variant'] not in set(observed.variant):
        observed=pd.concat([observed,pd.DataFrame([{'variant':start['variant'],'fitness':start['fitness']}])])
    if not observed.variant.is_unique:
        raise ValueError('Duplicate baseline history labels')
    return observed


def summary(branch,last):
    start=load_start()
    initial=pd.read_csv(ap.AGENT_INITIAL_FILE)
    best=max(float(initial.fitness.max()),float(start['fitness']))
    initial_best=best
    rows=[]
    for n in range(1,last+1):
        data=pd.read_csv(branch/f'round_{n:03d}/experiment_results.csv')
        best=max(best,float(data.fitness.max()))
        rows.append({'round':n,'n_measured':len(data),'round_best_fitness':float(data.fitness.max()),
                     'round_mean_fitness':float(data.fitness.mean()),'agent_best_so_far':best,
                     'initial_agent_best_fitness':initial_best,'count_above_wt':int(data.fitness.gt(1).sum()),
                     'cumulative_new_experiments':n*20,
                     'cumulative_total_experiments':len(initial)+start['seed_oracle_queries']+n*20,
                     'search_protocol':PROTOCOL,'selected_two_site':int(data.mutation_distance_to_anchor.eq(2).sum())})
    pd.DataFrame(rows).to_csv(branch/'evolution_summary.csv',index=False)


def run_method(method,args):
    branch=branch_path(method,args.seed)
    branch.mkdir(parents=True,exist_ok=True)
    start=load_start()
    metadata={'protocol':PROTOCOL,'method':method,'seed':args.seed,'input_sha256':start['input_sha256'],
              'selection_policy':'common-radius2-target6-no-kg-exclusive-slots','kg_access':False}
    meta=branch/'baseline_metadata.json'
    if meta.exists() and json.loads(meta.read_text(encoding='utf-8'))!=metadata:
        raise ValueError('Baseline metadata changed; use a new result directory')
    meta.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    predictions=pd.read_csv(ap.POOL_PREDICTIONS)
    heldout=set(pd.read_csv(ap.TEST_FILE).variant)
    for number in range(args.start_round,args.start_round+args.rounds):
        folder=branch/f'round_{number:03d}'
        if folder.exists():raise FileExistsError(f'Refusing to replace baseline round: {folder}')
        p=ap.round_paths(number)
        p.folder=folder
        p.knowledge_mode='off'
        observed=observed_history(branch,number)
        state=search_state(p,branch_dir=branch,observed=observed)
        pool=candidate_pool(predictions,state['parent_variants'],set(observed.variant),heldout)
        pool['purpose']='test'
        pool['design_method']='baseline'
        pool['reference_variant']=pool.apply(lambda r:r.anchor_variant if r.mutation_distance_to_anchor==1 else '',axis=1)
        order=None
        if method=='random':
            order=np.random.default_rng(np.random.SeedSequence([args.seed,number])).permutation(pool.variant).tolist()
        selected,targets=select_batch(pool,set(observed.variant),state['parent_variants'],ordered=order)
        assert targets[2]==state['two_site_target']
        selected['two_site_target']=targets[2]
        selected.to_csv(folder/'selected_candidates.csv',index=False)
        # Only selected IDs are joined to labels after the selection file exists.
        oracle=pd.read_csv(ap.ORACLE_FILE,usecols=['AACombo','Fitness']).rename(columns={'AACombo':'variant','Fitness':'fitness'})
        result=selected.merge(oracle,on='variant',how='left',validate='one_to_one')
        if result.fitness.isna().any():raise ValueError('Selected variant missing from Oracle')
        result['prediction_error']=result.predicted_fitness-result.fitness
        result.to_csv(folder/'experiment_results.csv',index=False)
        summary(branch,number)
        print(f'{method} round {number}: 20 experiments, {targets[2]} two-site, parents={state["parent_variants"]}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['both','random','m0'],default='both')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--start-round',type=int,default=1)
    parser.add_argument('--seed',type=int,default=114514)
    parser.add_argument('--start-sequence',default=None)
    args=parser.parse_args()
    if min(args.rounds,args.start_round)<1 or args.seed<0:parser.error('Invalid rounds or seed')
    os.environ['AGENT_KNOWLEDGE_MODE']='off'
    os.environ['GB1_KG_VALIDATION_SLOTS']='0'
    if not ap.AGENT_INITIAL_FILE.exists():
        command=[sys.executable,str(ap.AGENT_DIR/'0.initialize_agent_experiments.py')]
        if args.start_sequence:command+=['--start-sequence',args.start_sequence]
        subprocess.run(command,check=True)
    initialize_search(args.start_sequence)
    for method in (['random','m0'] if args.method=='both' else [args.method]):run_method(method,args)


if __name__=='__main__':main()
