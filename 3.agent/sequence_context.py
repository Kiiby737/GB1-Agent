"""Construct inference sequences on the actual M0 training background."""
import csv
import hashlib
import json
from pathlib import Path

POSITIONS = (39, 40, 41, 54)
AA = set('ACDEFGHIKLMNPQRSTVWY')
PREDICTION_SCHEMA = 'gb1-training-background-1'


def sequence_for_variant(variant, template):
    variant = str(variant).strip().upper()
    if len(variant) != 4 or not set(variant) <= AA:
        raise ValueError(f'Invalid GB1 variant: {variant}')
    sequence = list(template)
    if len(sequence) != 56 or not set(sequence) <= AA:
        raise ValueError('Expected a canonical 56-residue training template')
    for position, residue in zip(POSITIONS, variant):
        sequence[position - 1] = residue
    return ''.join(sequence)


def training_template(labeled_file):
    template = None
    with Path(labeled_file).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            sequence = row['mutant_seq'].strip().upper()
            variant = row['variant'].strip().upper()
            if sequence_for_variant(variant, sequence) != sequence:
                raise ValueError(f'M0 training sequence does not match variant {variant}')
            background = sequence_for_variant('VDGV', sequence)
            if template is not None and background != template:
                raise ValueError('M0 training data contain multiple sequence backgrounds')
            template = background
    if template is None:
        raise ValueError('M0 training labels are empty')
    return template


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prediction_metadata_path(path):
    return Path(path).with_suffix('.metadata.json')


def embedding_cache_path(base_file, rows, esm_name, positions):
    digest = hashlib.sha256(json.dumps([esm_name, list(positions)]).encode('utf-8'))
    for variant, sequence in rows:
        digest.update(f'{variant},{sequence}\n'.encode('utf-8'))
    base_file = Path(base_file)
    return base_file.with_name(f'{base_file.stem}_{digest.hexdigest()[:16]}.npy')


def write_prediction_metadata(path, labeled_file, model_file, esm_name):
    metadata = {
        'schema': PREDICTION_SCHEMA,
        'sequence_template': training_template(labeled_file),
        'prediction_sha256': file_sha256(path),
        'checkpoint_sha256': file_sha256(model_file),
        'esm_name': esm_name,
        'positions': list(POSITIONS),
    }
    prediction_metadata_path(path).write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')


def validate_prediction_pool(path, labeled_file, model_file):
    """Reject old scores even if somebody only relabelled their sequence column."""
    metadata_path = prediction_metadata_path(path)
    message = ('Rebuild predictions with 3.agent/0.prepare_M0_pool.py, '
               'then initialize a new GB1_AGENT_RESULT_DIR.')
    if not metadata_path.exists():
        raise ValueError(f'Prediction pool has no verified sequence-background metadata. {message}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    expected = {
        'schema': PREDICTION_SCHEMA,
        'sequence_template': training_template(labeled_file),
        'prediction_sha256': file_sha256(path),
        'checkpoint_sha256': file_sha256(model_file),
        'esm_name': 'esm2_t33_650M_UR50D',
        'positions': list(POSITIONS),
    }
    if metadata != expected:
        raise ValueError(f'Prediction pool background, checkpoint or file content changed. {message}')
