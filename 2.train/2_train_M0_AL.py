
from pathlib import Path
import copy
import random
import esm
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

FLIP_CSV = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4.6/1.data/sampled.csv')
INITIAL_FILE = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4.6/1.data/initial_96_labeled.csv')
SAVE_DIR = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4.6/2.train')
RUN_DIR = SAVE_DIR / 'hybrid_30_40_30_r9'

ESM_NAME = 'esm2_t33_650M_UR50D'
POSITIONS = [39, 40, 41, 54]
AA_TO_INT = {a: i for i, a in enumerate('ACDEFGHIKLMNPQRSTVWY')}
SEED = 114514
ESM_BATCH_SIZE = 4
PREDICT_BATCH_SIZE = 4096
N_MODELS = 5
HIDDEN_DIM = 32
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 500
PATIENCE = 50
AL_ROUNDS = 9
FITNESS_PICKS = 30
UNCERTAINTY_PICKS = 40
DIVERSITY_PICKS = 30
DIVERSITY_POOL_SIZE = 8000


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


def embed_train(train, device):
    cache = SAVE_DIR / 'flip_train_esm2_site4_float16.npy'
    if cache.exists():
        return np.load(cache, mmap_mode='r')

    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_NAME)
    model = model.to(device).eval()
    converter = alphabet.get_batch_converter()
    partial = SAVE_DIR / 'flip_train_esm2_site4_partial.npy'
    features = np.lib.format.open_memmap(
        partial, mode='w+', dtype=np.float16, shape=(len(train), 5120)
    )
    with torch.inference_mode():
        for start in range(0, len(train), ESM_BATCH_SIZE):
            batch = train.iloc[start:start + ESM_BATCH_SIZE]
            _, _, tokens = converter(list(zip(batch['variant'], batch['mutant_seq'])))
            rep = model(tokens.to(device), repr_layers=[model.num_layers],
                        return_contacts=False)['representations'][model.num_layers]
            features[start:start + len(batch)] = rep[:, POSITIONS, :].reshape(
                len(batch), -1
            ).cpu().numpy().astype(np.float16)
            if (start + len(batch)) % 400 == 0 or start + len(batch) == len(train):
                print(f'ESM2 train: {start + len(batch)}/{len(train)}', flush=True)
    features.flush()
    del features, model
    partial.replace(cache)
    return np.load(cache, mmap_mode='r')


def train_ensemble(features, measured, fitness_values, device):
    z = np.asarray(features[measured], dtype=np.float32)
    center = z.mean(axis=0)
    scale = z.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = torch.from_numpy((z - center) / scale).to(device)
    fitness_values = np.asarray(fitness_values, dtype=np.float32)
    members = []
    for i in range(N_MODELS):
        seed = SEED + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        train_idx, val_idx = train_test_split(
            np.arange(len(fitness_values)), test_size=0.1, random_state=seed
        )
        y_mean = float(fitness_values[train_idx].mean())
        y_std = float(fitness_values[train_idx].std())
        if y_std < 1e-8:
            y_std = 1.0
        y = torch.from_numpy((fitness_values - y_mean) / y_std).to(device)
        model = MLP(x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_loss = float('inf')
        best_state = None
        patience = 0
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x[train_idx]), y[train_idx])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = torch.nn.functional.mse_loss(model(x[val_idx]), y[val_idx]).item()
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= PATIENCE:
                break
        members.append({
            'model_state_dict': {k: v.cpu() for k, v in best_state.items()},
            'fitness_mean': y_mean,
            'fitness_std': y_std,
            'seed': seed,
        })
        print(f'MLP {i+1}/{N_MODELS}: val RMSE={np.sqrt(best_loss)*y_std:.4f}', flush=True)
    return {
        'esm2_name': ESM_NAME, 'positions': POSITIONS, 'embedding_mode': 'site4',
        'input_dim': int(z.shape[1]), 'hidden_dim': HIDDEN_DIM, 'dropout': DROPOUT,
        'feature_mean': torch.from_numpy(center.copy()),
        'feature_scale': torch.from_numpy(scale.copy()),
        'ensemble': members, 'n_training_sequences': len(fitness_values),
    }


def predict_pool(features, available, checkpoint, device):
    center = checkpoint['feature_mean'].to(device)
    scale = checkpoint['feature_scale'].to(device)
    models = []
    for member in checkpoint['ensemble']:
        mlp = MLP(checkpoint['input_dim']).to(device)
        mlp.load_state_dict(member['model_state_dict'])
        models.append((mlp.eval(), member['fitness_mean'], member['fitness_std']))
    scores = np.empty(len(available), dtype=np.float32)
    uncertainties = np.empty(len(available), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(available), PREDICT_BATCH_SIZE):
            indices = available[start:start + PREDICT_BATCH_SIZE]
            x = (torch.from_numpy(np.asarray(features[indices], dtype=np.float32)).to(device)
                 - center) / scale
            predictions = np.column_stack([
                mlp(x).cpu().numpy() * std + mean for mlp, mean, std in models
            ])
            scores[start:start + len(indices)] = predictions.mean(axis=1)
            uncertainties[start:start + len(indices)] = predictions.std(axis=1, ddof=1)
    return scores, uncertainties


def rank_diverse(indices, scores, k, codes, selected):
    ordered = indices[np.argsort(-scores)]
    chosen, nearby = [], []
    for idx in ordered:
        idx = int(idx)
        if selected or chosen:
            neighbors = codes[selected + chosen]
            if np.count_nonzero(neighbors != codes[idx], axis=1).min() < 2:
                nearby.append(idx)
                continue
        chosen.append(idx)
        if len(chosen) == k:
            break
    if len(chosen) < k:
        chosen.extend(nearby[:k - len(chosen)])
    return chosen


def pick_batch(available, scores, uncertainty, measured, codes, round_num):
    fitness = rank_diverse(available, scores, FITNESS_PICKS, codes, [])
    mask = ~np.isin(available, fitness)
    uncertain = rank_diverse(
        available[mask], uncertainty[mask], UNCERTAINTY_PICKS, codes, fitness
    )
    selected = fitness + uncertain
    remaining = available[~np.isin(available, selected)]
    rng = np.random.default_rng(SEED + round_num)
    sample = rng.choice(remaining, size=min(DIVERSITY_POOL_SIZE, len(remaining)),
                        replace=False)
    distances = np.full(len(sample), len(POSITIONS), dtype=np.int8)
    anchors = np.concatenate([measured, np.asarray(selected, dtype=np.int64)])
    for anchor in codes[anchors]:
        distances = np.minimum(distances, np.count_nonzero(codes[sample] != anchor, axis=1))
    diverse = []
    ties = rng.random(len(sample)) * 1e-3
    for _ in range(DIVERSITY_PICKS):
        j = int(np.argmax(distances + ties))
        idx = int(sample[j])
        diverse.append(idx)
        distances = np.minimum(distances, np.count_nonzero(codes[sample] != codes[idx], axis=1))
        distances[j] = -1
    return fitness + uncertain + diverse, (
        ['high_fitness'] * len(fitness) +
        ['high_uncertainty'] * len(uncertain) +
        ['diversity'] * len(diverse)
    )


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(FLIP_CSV, usecols=['sequence', 'target', 'set'])
    train = df.loc[df['set'].str.lower().str.strip() == 'train'].reset_index(drop=True)
    train['mutant_seq'] = train['sequence'].str.upper().str.strip().str[:56]
    train['variant'] = train['mutant_seq'].map(lambda s: ''.join(s[i] for i in (38, 39, 40, 53)))
    train['fitness'] = pd.to_numeric(train['target'])

    initial = pd.read_csv(INITIAL_FILE)
    labels = dict(zip(initial['variant'], initial['fitness']))
    features = embed_train(train, device)
    variants = train['variant'].tolist()
    index = {v: i for i, v in enumerate(variants)}
    codes = np.asarray([[AA_TO_INT[a] for a in s] for s in variants], dtype=np.uint8)
    history = []

    for round_num in range(1, AL_ROUNDS + 1):
        measured = np.asarray(sorted(index[v] for v in labels), dtype=np.int64)
        fitness_values = np.asarray([labels[variants[i]] for i in measured], dtype=np.float32)
        print(f'AL round {round_num}/{AL_ROUNDS}: training {len(measured)} labels', flush=True)
        checkpoint = train_ensemble(features, measured, fitness_values, device)
        torch.save(checkpoint, RUN_DIR / f'acquisition_model_round_{round_num - 1:02d}.pt')

        available = np.asarray([i for i, v in enumerate(variants) if v not in labels],
                               dtype=np.int64)
        scores, uncertainties = predict_pool(features, available, checkpoint, device)
        selected, strategies = pick_batch(
            available, scores, uncertainties, measured, codes, round_num
        )
        positions = np.searchsorted(available, selected)
        rows = train.iloc[selected][['variant', 'mutant_seq', 'fitness']].copy()
        rows.insert(0, 'round', round_num)
        rows.insert(1, 'strategy', strategies)
        rows['predicted_fitness'] = scores[positions]
        rows['uncertainty'] = uncertainties[positions]
        history.append(rows)
        pd.concat(history, ignore_index=True).to_csv(
            RUN_DIR / 'acquisition_history.csv', index=False
        )
        labels.update(zip(rows['variant'], rows['fitness']))
        print(f'Round {round_num}: acquired={len(selected)}, '
              f'best={rows.fitness.max():.4f}, labels={len(labels)}', flush=True)

    measured = np.asarray(sorted(index[v] for v in labels), dtype=np.int64)
    fitness_values = np.asarray([labels[variants[i]] for i in measured], dtype=np.float32)
    print(f'Final training with {len(measured)} labels', flush=True)
    final = train_ensemble(features, measured, fitness_values, device)
    torch.save(final, RUN_DIR / 'M0_AL_ensemble.pt')
    labeled = train.iloc[measured][['variant', 'mutant_seq']].copy()
    labeled['fitness'] = fitness_values
    labeled.to_csv(RUN_DIR / 'M0_AL_labeled.csv', index=False)

    remaining = np.asarray([i for i, v in enumerate(variants) if v not in labels],
                           dtype=np.int64)
    pool_mean, pool_std = predict_pool(features, remaining, final, device)
    pd.DataFrame({
        'variant': train.iloc[remaining]['variant'].to_numpy(),
        'mutant_seq': train.iloc[remaining]['mutant_seq'].to_numpy(),
        'predicted_fitness': pool_mean, 'uncertainty': pool_std,
    }).to_csv(RUN_DIR / 'pool_predictions.csv', index=False)
    print('Model:', RUN_DIR / 'M0_AL_ensemble.pt')
    print('Labels:', RUN_DIR / 'M0_AL_labeled.csv')
    print('Pool predictions:', RUN_DIR / 'pool_predictions.csv')


if __name__ == '__main__':
    main()
