"""FLIP GB1: 96 initial sequences, VER4 facility-location selection unchanged."""

from pathlib import Path
import esm
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

FLIP_CSV = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4/1.data/sampled.csv')
SAVE_DIR = Path('/mnt/disk2/shiming/project/ZJU_agent_project/ver4/1.data')

ESM_NAME = 'esm2_t33_650M_UR50D'
POSITIONS = [39, 40, 41, 54]
WT_SEQ = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INT = {a: i for i, a in enumerate(AA_ORDER)}
N_SELECT = 96
N_ANCHORS = 2048
N_PROPOSALS = 8192
ESM_PRIOR_STRENGTH = 0.10
SINGLE_WEIGHT = 0.75
PAIR_WEIGHT = 0.25
SEED = 114514


AA_PROPERTIES = {
    'A': (89.094, 1.8, 115.0, 88.6),
    'R': (174.203, -4.5, 225.0, 173.4),
    'D': (133.104, -3.5, 150.0, 111.1),
    'N': (132.119, -3.5, 160.0, 114.1),
    'C': (121.154, 2.5, 135.0, 108.5),
    'E': (147.131, -3.5, 190.0, 138.4),
    'Q': (146.146, -3.5, 180.0, 143.8),
    'G': (75.067, -0.4, 75.0, 60.1),
    'H': (155.156, -3.2, 195.0, 153.2),
    'I': (131.175, 4.5, 175.0, 166.7),
    'L': (131.175, 3.8, 170.0, 166.7),
    'K': (146.189, -3.9, 200.0, 168.6),
    'M': (149.208, 1.9, 185.0, 162.9),
    'F': (165.192, 2.8, 210.0, 189.9),
    'P': (115.132, -1.6, 145.0, 112.7),
    'S': (105.093, -0.8, 115.0, 89.0),
    'T': (119.120, -0.7, 140.0, 116.1),
    'W': (204.228, -0.9, 255.0, 227.8),
    'Y': (181.191, -1.3, 230.0, 193.6),
    'V': (117.148, 4.2, 155.0, 140.0),
}


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FLIP_CSV, usecols=['sequence', 'target', 'set'])
    df = df.loc[df['set'].str.lower().str.strip() == 'train'].reset_index(drop=True)
    df['mutant_seq'] = df['sequence'].str.upper().str.strip().str[:56]
    df['variant'] = df['mutant_seq'].map(lambda s: ''.join(s[i] for i in (38, 39, 40, 53)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_NAME)
    model = model.to(device).eval()
    _, _, tokens = alphabet.get_batch_converter()([('WT', WT_SEQ)])
    tokens[0, POSITIONS] = alphabet.mask_idx
    with torch.inference_mode():
        logits = model(tokens.to(device), return_contacts=False)['logits'][0]
        ids = [alphabet.get_idx(a) for a in AA_ORDER]
        log_probs = torch.log_softmax(logits[POSITIONS][:, ids].float(), dim=-1).cpu().numpy()
    del model

    variants = np.asarray([[AA_TO_INT[a] for a in s] for s in df['variant']], dtype=np.int16)
    raw_score = log_probs[np.arange(4)[None, :], variants].mean(axis=1)
    quality = pd.Series(raw_score).rank(pct=True).to_numpy(dtype=np.float32)

    charge = {'D': -1, 'E': -1, 'K': 1, 'R': 1}
    polar = set('DEHKNQRSTY')
    aromatic = set('FHWY')
    aa_features = np.asarray([
        [AA_PROPERTIES[a][1], AA_PROPERTIES[a][3], charge.get(a, 0),
         float(a in polar), float(a in aromatic), float(a == 'G'), float(a == 'P')]
        for a in AA_ORDER
    ], dtype=np.float32)
    aa_features = StandardScaler().fit_transform(aa_features).astype(np.float32)
    site_features = aa_features[variants]
    single = site_features.reshape(len(df), -1)
    pair_parts = []
    for i in range(4):
        for j in range(i + 1, 4):
            pair_parts.append(site_features[:, i, :5] * site_features[:, j, :5])
    pair = np.concatenate(pair_parts, axis=1)
    single = StandardScaler().fit_transform(single).astype(np.float32)
    pair = StandardScaler().fit_transform(pair).astype(np.float32)
    features = np.concatenate([
        np.sqrt(SINGLE_WEIGHT / single.shape[1]) * single,
        np.sqrt(PAIR_WEIGHT / pair.shape[1]) * pair,
    ], axis=1)

    rng = np.random.default_rng(SEED)
    n = len(df)
    anchors = rng.choice(n, size=min(N_ANCHORS, n), replace=False)
    proposals = rng.choice(n, size=min(N_PROPOSALS, n), replace=False)
    proposals = np.unique(np.concatenate([proposals, anchors]))
    x, z = features[proposals], features[anchors]
    squared_dist = (np.sum(x * x, axis=1)[:, None]
                    + np.sum(z * z, axis=1)[None, :]
                    - 2 * (x @ z.T))
    np.maximum(squared_dist, 0, out=squared_dist)
    ia = rng.integers(0, len(anchors), 4096)
    ib = rng.integers(0, len(anchors), 4096)
    sigma2 = float(np.median(np.sum((z[ia] - z[ib]) ** 2, axis=1)))
    similarities = np.exp(-squared_dist / (2 * sigma2)).astype(np.float32)
    similarities *= (1 + ESM_PRIOR_STRENGTH * quality[proposals, None])

    covered = np.zeros(len(anchors), dtype=np.float32)
    selected = []
    for _ in range(N_SELECT):
        gain = np.maximum(similarities - covered[None, :], 0).mean(axis=1)
        gain[selected] = -np.inf
        i = int(np.argmax(gain))
        selected.append(i)
        covered = np.maximum(covered, similarities[i])

    chosen = df.iloc[proposals[selected]]['variant'].tolist()
    selected_df = df.set_index('variant').loc[chosen, ['mutant_seq', 'target']].reset_index()
    selected_df[['variant', 'mutant_seq']].to_csv(SAVE_DIR / 'initial_96_candidates.csv', index=False)
    selected_df.rename(columns={'target': 'fitness'}).to_csv(
        SAVE_DIR / 'initial_96_labeled.csv', index=False
    )
    print(f'Selected {len(selected_df)} initial sequences: {SAVE_DIR / "initial_96_labeled.csv"}')


if __name__ == '__main__':
    main()
