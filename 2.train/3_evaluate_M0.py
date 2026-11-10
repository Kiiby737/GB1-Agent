from pathlib import Path
import esm
import numpy as np
import pandas as pd
import torch

FLIP_CSV = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4.6/1.data/sampled.csv')
SAVE_DIR = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4.6/2.train')
RUN_DIR = SAVE_DIR / 'hybrid_30_40_30_r9'
ESM_NAME = 'esm2_t33_650M_UR50D'
POSITIONS = [39, 40, 41, 54]
HIDDEN_DIM = 32
DROPOUT = 0.1
ESM_BATCH_SIZE = 4
PREDICT_BATCH_SIZE = 4096


class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def embed_test(test, device):
    cache = SAVE_DIR / 'flip_test_esm2_site4_float16.npy'
    if cache.exists():
        return np.load(cache, mmap_mode='r')

    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_NAME)
    model = model.to(device).eval()
    converter = alphabet.get_batch_converter()
    partial = SAVE_DIR / 'flip_test_esm2_site4_partial.npy'
    features = np.lib.format.open_memmap(
        partial, mode='w+', dtype=np.float16, shape=(len(test), 5120)
    )
    with torch.inference_mode():
        for start in range(0, len(test), ESM_BATCH_SIZE):
            batch = test.iloc[start:start + ESM_BATCH_SIZE]
            _, _, tokens = converter(list(zip(batch['variant'], batch['mutant_seq'])))
            rep = model(tokens.to(device), repr_layers=[model.num_layers],
                        return_contacts=False)['representations'][model.num_layers]
            features[start:start + len(batch)] = rep[:, POSITIONS, :].reshape(
                len(batch), -1
            ).cpu().numpy().astype(np.float16)
            if (start + len(batch)) % 400 == 0 or start + len(batch) == len(test):
                print(f'ESM2 test: {start + len(batch)}/{len(test)}', flush=True)
    features.flush()
    del features, model
    partial.replace(cache)
    return np.load(cache, mmap_mode='r')


def predict(features, checkpoint, device):
    center = checkpoint['feature_mean'].to(device)
    scale = checkpoint['feature_scale'].to(device)
    models = []
    for member in checkpoint['ensemble']:
        mlp = MLP(checkpoint['input_dim']).to(device)
        mlp.load_state_dict(member['model_state_dict'])
        models.append((mlp.eval(), member['fitness_mean'], member['fitness_std']))

    scores = np.empty(len(features), dtype=np.float32)
    uncertainties = np.empty(len(features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(features), PREDICT_BATCH_SIZE):
            x = (torch.from_numpy(np.asarray(
                features[start:start + PREDICT_BATCH_SIZE], dtype=np.float32
            )).to(device) - center) / scale
            predictions = np.column_stack([
                mlp(x).cpu().numpy() * std + mean for mlp, mean, std in models
            ])
            scores[start:start + len(predictions)] = predictions.mean(axis=1)
            uncertainties[start:start + len(predictions)] = predictions.std(axis=1, ddof=1)
    return scores, uncertainties


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(FLIP_CSV, usecols=['sequence', 'target', 'set'])
    test = df.loc[df['set'].str.lower().str.strip() == 'test'].reset_index(drop=True)
    test['mutant_seq'] = test['sequence'].str.upper().str.strip().str[:56]
    test['variant'] = test['mutant_seq'].map(lambda s: ''.join(s[i] for i in (38, 39, 40, 53)))
    test['fitness'] = pd.to_numeric(test['target'])
    features = embed_test(test, device)
    y = test['fitness'].to_numpy(dtype=np.float32)
    true_top = np.argsort(-y, kind='stable')

    results = []
    for stage in [0, 1, 2, 3, 9]:
        name = f'AL_{stage:02d}' if stage < 9 else 'AL_Final'
        path = (RUN_DIR / f'acquisition_model_round_{stage:02d}.pt'
                if stage < 9 else RUN_DIR / 'M0_AL_ensemble.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        pred, uncertainty = predict(features, checkpoint, device)
        error = pred - y
        predicted_top = np.argsort(-pred, kind='stable')
        metrics = {
            'stage': name,
            'n_labels': 96 + stage * 100 if stage < 9 else 496,
            'MAE': float(np.abs(error).mean()),
            'MSE': float(np.square(error).mean()),
            'RMSE': float(np.sqrt(np.square(error).mean())),
            'Pearson': float(np.corrcoef(y, pred)[0, 1]),
            'Spearman': float(pd.Series(y).corr(pd.Series(pred), method='spearman')),
        }
        for k in (5, 10, 20):
            metrics[f'Top{k}_overlap'] = len(set(true_top[:k]) & set(predicted_top[:k]))
        results.append(metrics)
        print(f"{name} | N={metrics['n_labels']} | Pearson={metrics['Pearson']:.4f} | "
              f"Spearman={metrics['Spearman']:.4f} | Top10={metrics['Top10_overlap']}/10")
        if stage == 9:
            pd.DataFrame({
                'variant': test['variant'],
                'mutant_seq': test['mutant_seq'],
                'fitness': test['fitness'],
                'predicted_fitness': pred,
            }).to_csv(RUN_DIR / 'test_model_predictions.csv', index=False)
    pd.DataFrame(results).to_csv(RUN_DIR / 'test_model_metrics.csv', index=False)
    print('Metrics:', RUN_DIR / 'test_model_metrics.csv')


if __name__ == '__main__':
    main()
