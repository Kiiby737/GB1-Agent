import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Added: plotting dependencies
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'3.agent'))
import agent_paths as ap
from search_strategy import PROTOCOL,load_start
from common_search import candidate_pool,domain_hash,domain,allocation


def collect(rounds,seed):
    start=load_start()
    initial=pd.read_csv(ap.AGENT_INITIAL_FILE)
    model_history=pd.read_csv(ap.INITIAL_FILE)[['variant','fitness']]
    predictions=pd.read_csv(ap.POOL_PREDICTIONS)
    scores=predictions.set_index('variant').predicted_fitness
    truth=pd.read_csv(ap.ORACLE_FILE,usecols=['AACombo','Fitness']).set_index('AACombo').Fitness
    heldout=set(pd.read_csv(ap.TEST_FILE).variant)
    sources={'Random Search':ap.AGENT_RESULT_DIR/'baselines'/f'random_search_seed{seed}',
             'M0-AL Only':ap.AGENT_RESULT_DIR/'baselines/m0_only',
             'LLM Agent OFF':ap.BASE_ROUNDS_DIR,'LLM Agent ON':ap.KG_ROUNDS_DIR}
    rows=[]
    for name,branch in sources.items():
        observed=pd.concat([model_history,initial[['variant','fitness']]])
        if start['variant'] not in set(observed.variant):
            observed=pd.concat([observed,pd.DataFrame([{'variant':start['variant'],'fitness':start['fitness']}])])
        lineage={start['variant']}
        best=max(float(initial.fitness.max()),float(start['fitness']))
        for n in range(1,rounds+1):
            folder=branch/f'round_{n:03d}'
            state=json.loads((folder/'search_state.json').read_text(encoding='utf-8'))
            data=pd.read_csv(folder/'experiment_results.csv',keep_default_na=False)
            if state['protocol']!=PROTOCOL or state['selection_policy']!='common-radius2-target6-no-kg-exclusive-slots':
                raise ValueError('Mixed experiment protocols')
            if not set(state['parent_variants'])<=lineage:raise ValueError('Parent outside measured seed lineage')
            eligible=set(predictions.variant)-set(observed.variant)-heldout
            assigned=domain(state['parent_variants'],eligible)
            if domain_hash(assigned)!=state['admissible_pool_sha256']:raise ValueError('Search-domain mismatch')
            if len(data)!=20 or not data.variant.is_unique:raise ValueError('Unequal/repeated experiment budget')
            if not np.allclose(data.predicted_fitness,data.variant.map(scores),rtol=0,atol=1e-8):
                raise ValueError('Results use a different prediction pool')
            if not np.allclose(data.fitness,data.variant.map(truth),rtol=0,atol=1e-10):
                raise ValueError('Saved experiment labels differ from the Oracle')
            for item in data.itertuples():
                if assigned.get(item.variant)!=(item.anchor_variant,int(item.mutation_distance_to_anchor)):
                    raise ValueError('Result outside the common canonical search domain')
            if int(data.mutation_distance_to_anchor.eq(2).sum())!=state['two_site_target']:
                raise ValueError('Mutation-order allocation differs from the common policy')
            best=max(best,float(data.fitness.max()))
            rows.append({'method':name,'round':n,'n_measured':20,'round_best_fitness':float(data.fitness.max()),
                         'round_mean_fitness':float(data.fitness.mean()),'agent_best_so_far':best,
                         'initial_agent_best_fitness':max(float(initial.fitness.max()),float(start['fitness'])),
                         'cumulative_total_experiments':len(initial)+start['seed_oracle_queries']+n*20,
                         'selected_two_site':int(data.mutation_distance_to_anchor.eq(2).sum()),'search_protocol':PROTOCOL})
            observed=pd.concat([observed,data[['variant','fitness']]])
            lineage.update(data.variant)
    return pd.DataFrame(rows)


# Added: two-panel comparison figure (PDF + SVG vector exports).
def plot_comparison(table, rounds, output):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    methods = ['Random Search', 'M0-AL Only', 'LLM Agent OFF', 'LLM Agent ON']
    panels = [
        ('round_best_fitness', '(A) Best Fitness', 'Best Fitness'),
        ('round_mean_fitness', '(B) Mean Fitness', 'Mean Fitness'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, (column, title, ylabel) in zip(axes, panels):
        for method in methods:
            group = table.loc[table['method'].eq(method)].sort_values('round')
            ax.plot(group['round'], group[column], marker='o', markersize=6,
                    linewidth=1.8, label=method)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Evolution Round', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(range(1, rounds + 1))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(fontsize=9, frameon=False, loc='best')

    fig.tight_layout()
    pdf_path = output / 'four_method_comparison.pdf'
    svg_path = output / 'four_method_comparison.svg'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Plot PDF: {pdf_path}')
    print(f'Plot SVG: {svg_path}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--seed',type=int,default=114514)
    args=parser.parse_args()
    if args.rounds<1:parser.error('rounds must be positive')
    table=collect(args.rounds,args.seed)
    output=ap.AGENT_RESULT_DIR/'baselines/comparison'
    output.mkdir(parents=True,exist_ok=True)
    table.to_csv(output/'four_method_per_round.csv',index=False)
    summary=table.groupby('method',sort=False).agg(cumulative_best_fitness=('agent_best_so_far','max'),
                 all_rounds_mean_fitness=('round_mean_fitness','mean'),total_labels=('cumulative_total_experiments','max'))
    summary.to_csv(output/'four_method_summary.csv')
    print(summary.to_string())
    # Added: render figure after writing the original CSV results.
    plot_comparison(table, args.rounds, output)


if __name__=='__main__':main()
