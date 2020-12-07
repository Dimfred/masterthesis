from pathlib import Path

# DATA_DIR = Path('../data')
# DATA_LFW_DIR = DATA_DIR / 'lfw'

TRAIN_DIR = Path("../data/train_preprocessed")
VALID_DIR = Path("../data/valid_preprocessed")

EXP_DIR = Path('../experiments')

TMP_DIR = Path('../tmp')
TMP_DIR.mkdir(exist_ok=True)
