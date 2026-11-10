"""Score the original GB1 candidate pool with frozen VER4 M0-AL (no Oracle labels)."""
import numpy as np
import pandas as pd
import torch

from agent_paths import (
    MODEL_FILE, INITIAL_FILE, TEST_FILE, FLIP_FILE, ORACLE_FILE,
    EMBEDDING_CACHE, POOL_PREDICTIONS,
)
from sequence_context import (
    training_template, sequence_for_variant, write_prediction_metadata, embedding_cache_path,
)

ESM_NAME = 'esm2_t33_650M_UR50D'
POSITIONS = [39, 40, 41, 54]
ESM_BATCH_SIZE = 4
PREDICT_BATCH_SIZE = 4096


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def embed_pool(pool, device):
    # Sequence content/order and extraction settings identify the cache. The old
    # fixed-name cache may have the same row count but a different background.
    cache = embedding_cache_path(
        EMBEDDING_CACHE, zip(pool['variant'], pool['mutant_seq']), ESM_NAME, POSITIONS)
    if cache.exists():
        return np.load(cache, mmap_mode='r')

    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_NAME)
    model = model.to(device).eval()
    converter = alphabet.get_batch_converter()
    partial = cache.with_suffix('.partial.npy')
    partial.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        partial, mode='w+', dtype=np.float16, shape=(len(pool), 5120)
    )
    with torch.inference_mode():
        for start in range(0, len(pool), ESM_BATCH_SIZE):
            batch = pool.iloc[start:start + ESM_BATCH_SIZE]
            _, _, tokens = converter(list(zip(batch['variant'], batch['mutant_seq'])))
            rep = model(tokens.to(device), repr_layers=[model.num_layers],
                        return_contacts=False)['representations'][model.num_layers]
            features[start:start + len(batch)] = rep[:, POSITIONS, :].reshape(
                len(batch), -1
            ).cpu().numpy().astype(np.float16)
            if (start + len(batch)) % 400 == 0 or start + len(batch) == len(pool):
                features.flush()
                print(f'ESM2 original GB1: {start + len(batch)}/{len(pool)}', flush=True)
    features.flush()
    del features, model
    partial.replace(cache)
    return np.load(cache, mmap_mode='r')


def main():
    # Retain the original VER4 held-out FLIP test partition.
    flip = pd.read_csv(FLIP_FILE, usecols=['sequence', 'set'])
    flip_test = flip.loc[flip['set'].str.lower().str.strip() == 'test', 'sequence']
    test = pd.DataFrame({
        'variant': flip_test.str.upper().str.strip().map(
            lambda s: ''.join(s[i] for i in (38, 39, 40, 53))
        )
    })
    TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(TEST_FILE, index=False)

    labeled = set(pd.read_csv(INITIAL_FILE, usecols=['variant'])['variant'])
    raw = pd.read_csv(ORACLE_FILE, usecols=['AACombo']).rename(columns={'AACombo': 'variant'})
    template = training_template(INITIAL_FILE)
    raw['variant'] = raw['variant'].astype(str).str.upper().str.strip()
    raw['mutant_seq'] = raw['variant'].map(lambda v: sequence_for_variant(v, template))
    pool = raw.loc[~raw['variant'].isin(test['variant'])].reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = embed_pool(pool, device)
    assert len(features) == len(pool), 'Original GB1 ESM2 cache does not match pool length'
    available = np.flatnonzero(~pool['variant'].isin(labeled).to_numpy())

    checkpoint = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)
    center = checkpoint['feature_mean'].to(device)
    scale = checkpoint['feature_scale'].to(device)
    models = []
    for member in checkpoint['ensemble']:
        mlp = MLP(checkpoint['input_dim'], checkpoint['hidden_dim'], checkpoint['dropout']).to(device)
        mlp.load_state_dict(member['model_state_dict'])
        models.append((mlp.eval(), member['fitness_mean'], member['fitness_std']))

    predictions_mean = np.empty(len(available), dtype=np.float32)
    predictions_std = np.empty(len(available), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(available), PREDICT_BATCH_SIZE):
            indices = available[start:start + PREDICT_BATCH_SIZE]
            z = torch.from_numpy(np.asarray(features[indices], dtype=np.float32)).to(device)
            x = (z - center) / scale
            pred = np.column_stack([
                mlp(x).cpu().numpy() * std + mean
                for mlp, mean, std in models
            ])
            predictions_mean[start:start + len(indices)] = pred.mean(axis=1)
            predictions_std[start:start + len(indices)] = pred.std(axis=1, ddof=1)

    result = pool.iloc[available][['variant', 'mutant_seq']].copy()
    result['predicted_fitness'] = predictions_mean
    result['uncertainty'] = predictions_std
    POOL_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(POOL_PREDICTIONS, index=False)
    write_prediction_metadata(POOL_PREDICTIONS, INITIAL_FILE, MODEL_FILE, ESM_NAME)
    print(f'Frozen VER4 M0-AL scored {len(result)} unseen original GB1 variants: {POOL_PREDICTIONS}')


if __name__ == '__main__':
    main()
