"""FlowCIR local paths. Edit here or override with FLOWCIR_* environment variables.

CLI arguments override these defaults. No datasets or checkpoints are downloaded.
"""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CLIP_MODEL = os.environ.get('FLOWCIR_CLIP_MODEL', 'ViT-L/14')
DATA_ROOT = Path(os.environ.get('FLOWCIR_DATA_ROOT', PROJECT_ROOT / 'data')).expanduser()
FEATURE_ROOT = Path(os.environ.get('FLOWCIR_FEATURE_ROOT', PROJECT_ROOT / 'features')).expanduser()
HQEDIT_DIR = Path(os.environ.get('FLOWCIR_HQEDIT_DIR', DATA_ROOT / 'HQ-Edit')).expanduser()
HQEDIT_FEATURE_DIR = Path(
    os.environ.get('FLOWCIR_HQEDIT_FEATURE_DIR', FEATURE_ROOT / 'hqedit')
).expanduser()
CIRR_DIR = Path(os.environ.get('FLOWCIR_CIRR_DIR', DATA_ROOT / 'cirr')).expanduser()
CIRCO_DIR = Path(os.environ.get('FLOWCIR_CIRCO_DIR', DATA_ROOT / 'CIRCO')).expanduser()
SAVE_ROOT = Path(os.environ.get('FLOWCIR_SAVE_ROOT', PROJECT_ROOT / 'checkpoints')).expanduser()
LOG_DIR = Path(os.environ.get('FLOWCIR_LOG_DIR', PROJECT_ROOT / 'logs')).expanduser()
SUBMISSION_DIR = Path(os.environ.get('FLOWCIR_SUBMISSION_DIR', PROJECT_ROOT / 'submissions')).expanduser()
CLIP_DIMS = {'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768, 'ViT-L/14@336px': 768}


def feature_path(dataset, split, kind, clip_model=None):
    if dataset not in ('cirr', 'circo') or split not in ('val', 'test') or kind not in ('query', 'gallery'):
        raise ValueError('Expected cirr/circo, val/test, query/gallery')
    model = (clip_model or CLIP_MODEL).replace('/', '-').replace('@', '-')
    return FEATURE_ROOT / dataset / model / f'{split}_{kind}.pth'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('key', nargs='?')
    parser.add_argument('--feature', nargs=3, metavar=('DATASET', 'SPLIT', 'KIND'))
    args = parser.parse_args()
    if args.feature:
        print(feature_path(*args.feature))
    elif args.key == 'DIM':
        print(CLIP_DIMS[CLIP_MODEL])
    elif args.key in ('PROJECT_ROOT', 'DATA_ROOT', 'FEATURE_ROOT', 'HQEDIT_DIR', 'HQEDIT_FEATURE_DIR',
                      'CIRR_DIR', 'CIRCO_DIR', 'SAVE_ROOT', 'LOG_DIR', 'SUBMISSION_DIR', 'CLIP_MODEL'):
        print(globals()[args.key])
    else:
        parser.error('Specify a config key or --feature DATASET SPLIT KIND')
