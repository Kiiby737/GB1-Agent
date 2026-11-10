"""Evaluate frozen M0 on sampled.csv test sequences, using its training background."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
# Deployment layout remains ver4.8/3.agent and ver4.8/5.test.
sys.path.insert(0, str(ROOT / '3.agent'))
from agent_paths import FLIP_FILE, INITIAL_FILE, MODEL_FILE, TEST_FILE
from sequence_context import training_template, sequence_for_variant, file_sha256

OUT_DIR = Path(__file__).resolve().parent / 'm0_al_independent_test'
ESM_NAME = 'esm2_t33_650M_UR50D'
POSITIONS = (39, 40, 41, 54)
SCHEMA = 'm0-sampled-test-background-1'
AA = set('ACDEFGHIKLMNPQRSTVWY')


def load_test_data():
    """Use exactly the sequence[:56], target and official split used by training."""
    sampled = pd.read_csv(FLIP_FILE, usecols=['sequence', 'target', 'set'])
    sampled['seq'] = sampled.sequence.astype(str).str.upper().str.strip().str[:56]
    sampled['split'] = sampled['set'].astype(str).str.lower().str.strip()
    if not sampled.seq.map(lambda s: len(s) == 56 and set(s) <= AA).all():
        raise ValueError('sampled.csv contains a noncanonical or short GB1 sequence')
    sampled['variant'] = sampled.seq.map(lambda s: ''.join(s[p-1] for p in POSITIONS))
    if not sampled.variant.is_unique:
        raise ValueError('sampled.csv contains duplicate four-site variant IDs')
    template = training_template(INITIAL_FILE)
    expected = sampled.variant.map(lambda v: sequence_for_variant(v, template))
    bad = sampled.loc[sampled.seq.ne(expected), 'variant']
    if len(bad):
        raise ValueError(f'sampled.csv sequence background differs from M0 training: {bad.iloc[0]}')
    test = sampled.loc[sampled.split.eq('test'), ['variant', 'seq', 'target']].copy()
    if len(test) < 2:
        raise ValueError('sampled.csv must contain at least two held-out test variants')
    test = test.rename(columns={'target': 'fitness'})
    test['fitness'] = pd.to_numeric(test.fitness, errors='raise')
    if not np.isfinite(test.fitness).all():
        raise ValueError('Nonfinite sampled.csv test labels')
    trained = set(pd.read_csv(INITIAL_FILE, usecols=['variant']).variant.astype(str).str.upper().str.strip())
    train_ids = set(sampled.loc[sampled.split.eq('train'), 'variant'])
    if trained & set(test.variant) or not trained <= train_ids:
        raise ValueError('M0 training IDs overlap test or do not belong to sampled.csv train split')
    heldout = pd.read_csv(TEST_FILE, usecols=['variant'])
    heldout['variant'] = heldout.variant.astype(str).str.upper().str.strip()
    if not heldout.variant.is_unique:
        raise ValueError('Duplicate held-out test IDs')
    if set(heldout.variant) != set(test.variant):
        raise ValueError('sampled.csv test split differs from flip_test_variants.csv')
    # Preserve test-ID order explicitly; joins must not silently drop rows.
    test = heldout.merge(test, on='variant', how='left', validate='one_to_one', sort=False)
    return test, template


def input_metadata(test, template):
    rows = '\n'.join(f'{v},{s}' for v, s in test[['variant', 'seq']].itertuples(index=False, name=None))
    return {'schema': SCHEMA, 'sampled_sha256': file_sha256(FLIP_FILE),
            'training_labels_sha256': file_sha256(INITIAL_FILE),
            'checkpoint_sha256': file_sha256(MODEL_FILE), 'heldout_ids_sha256': file_sha256(TEST_FILE),
            'sequence_sha256': hashlib.sha256(rows.encode('utf-8')).hexdigest(),
            'sequence_template': template, 'n_test': len(test),
            'esm_name': ESM_NAME, 'positions': list(POSITIONS)}


def metadata_path(path):
    return Path(path).with_suffix('.metadata.json')


def validate_predictions(predictions, test):
    predictions = predictions[['variant', 'predicted_fitness']].copy()
    predictions['variant'] = predictions.variant.astype(str).str.upper().str.strip()
    if not predictions.variant.is_unique:
        raise ValueError('Duplicate variant IDs in independent-test predictions')
    if set(predictions.variant) != set(test.variant):
        raise ValueError('Predictions must cover exactly the complete sampled.csv test set')
    predictions['predicted_fitness'] = pd.to_numeric(predictions.predicted_fitness, errors='raise')
    if not np.isfinite(predictions.predicted_fitness).all():
        raise ValueError('Nonfinite independent-test predictions')
    return test.merge(predictions, on='variant', how='left', validate='one_to_one', sort=False)


def load_verified_predictions(path, expected):
    sidecar = metadata_path(path)
    if not sidecar.exists():
        raise ValueError('Predictions lack verified input-background metadata; rerun without --predictions')
    actual = json.loads(sidecar.read_text(encoding='utf-8'))
    if actual != {**expected, 'prediction_sha256': file_sha256(path)}:
        raise ValueError('Prediction sequence background, sampled data, model or file changed; recompute predictions')
    return pd.read_csv(path, usecols=['variant', 'predicted_fitness'])


def predict_m0(test, sequences_file, batch_size):
    import torch
    import esm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences = test.merge(sequences_file, on='variant', how='left', validate='one_to_one', sort=False)
    if sequences.seq.isna().any():
        raise ValueError('Missing sampled.csv test sequences')
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_NAME)
    esm_model = esm_model.to(device).eval()
    converter = alphabet.get_batch_converter()
    features = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences.iloc[start:start+batch_size]
            _, _, tokens = converter(list(zip(batch.variant, batch.seq)))
            reps = esm_model(tokens.to(device), repr_layers=[esm_model.num_layers],
                             return_contacts=False)['representations'][esm_model.num_layers]
            features.append(reps[:, POSITIONS, :].reshape(len(batch), -1).cpu())
            print(f'ESM2: {min(start+batch_size, len(sequences))}/{len(sequences)}', flush=True)
    del esm_model
    features = torch.cat(features).to(device)

    class MLP(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, dropout):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
                                           torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    checkpoint = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)
    x = (features-checkpoint['feature_mean'].to(device))/checkpoint['feature_scale'].to(device)
    member_predictions = []
    with torch.inference_mode():
        for member in checkpoint['ensemble']:
            model = MLP(checkpoint['input_dim'], checkpoint['hidden_dim'], checkpoint['dropout']).to(device)
            model.load_state_dict(member['model_state_dict'])
            model.eval()
            member_predictions.append(model(x).cpu().numpy()*member['fitness_std']+member['fitness_mean'])
    return pd.DataFrame({'variant': sequences.variant, 'predicted_fitness': np.mean(member_predictions, axis=0)})


def evaluate(result):
    y, yhat = result.fitness.to_numpy(float), result.predicted_fitness.to_numpy(float)
    mse = float(np.mean((y-yhat)**2))
    varying = np.ptp(y) > 0 and np.ptp(yhat) > 0
    metrics = {'n_test': len(result), 'pearson': float(np.corrcoef(y, yhat)[0,1]) if varying else None,
               'spearman': float(spearmanr(y, yhat)[0]) if varying else None,
               'mse': mse, 'rmse': float(np.sqrt(mse))}
    predicted_rank = result.sort_values(['predicted_fitness','variant'], ascending=[False,True])
    true_rank = result.sort_values(['fitness','variant'], ascending=[False,True])
    topk = []
    for k in (5,10):
        used = min(k,len(result))
        n = len(set(predicted_rank.variant.head(used)) & set(true_rank.variant.head(used)))
        topk.append({'k': k, 'evaluated_k': used, 'topk_overlap_count': n, 'topk_overlap_rate': n/used})
    return metrics, pd.DataFrame(topk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=None, help='Reuse only predictions with matching metadata')
    parser.add_argument('--esm-batch-size', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.esm_batch_size < 1:
        parser.error('esm-batch-size must be positive')
    test, template = load_test_data()
    provenance = input_metadata(test, template)
    if args.predictions:
        predictions = load_verified_predictions(args.predictions, provenance)
    else:
        # Neither labels nor Oracle sequences enter the inference function.
        predictions = predict_m0(test[['variant']], test[['variant','seq']], args.esm_batch_size)
    result = validate_predictions(predictions, test)
    metrics, topk = evaluate(result)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if not args.predictions:
        path = output/'test_predictions.csv'
        predictions[['variant','predicted_fitness']].to_csv(path, index=False)
        metadata_path(path).write_text(json.dumps({**provenance,'prediction_sha256':file_sha256(path)},
                                                indent=2), encoding='utf-8')
    result.rename(columns={'seq':'mutant_seq'}).to_csv(output/'m0_al_test_predictions_with_labels.csv',index=False)
    topk.to_csv(output/'m0_al_test_topk_metrics.csv',index=False)
    report = {**metrics, 'topk':topk.to_dict('records'), 'sequence_source':str(FLIP_FILE),
              'label_source':'sampled.csv:target', 'model_source':str(MODEL_FILE), 'input_provenance':provenance}
    (output/'m0_al_test_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2,5.2))
    ax.scatter(result.fitness,result.predicted_fitness,s=12,alpha=.35)
    lo=min(result.fitness.min(),result.predicted_fitness.min())
    hi=max(result.fitness.max(),result.predicted_fitness.max())
    ax.plot([lo,hi],[lo,hi],'--',linewidth=1)
    ax.set(xlabel='True Fitness',ylabel='Predicted Fitness',title='M0-AL Independent Test')
    fig.tight_layout()
    fig.savefig(output/'m0_al_test_scatter.png',dpi=180)
    plt.close(fig)
    print(json.dumps(metrics,indent=2,allow_nan=False))
    print(f'Sequences and labels: {FLIP_FILE}; training background verified; test count={len(test)}')
    print(f'Output: {output}')


if __name__=='__main__':
    main()
