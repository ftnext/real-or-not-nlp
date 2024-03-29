from pathlib import Path

EXPERIMENT_NUMBER = "001"

ROOT_DIR = Path(__file__).parent.parent
EXPERIMENT_DIR = ROOT_DIR / "experiments"

DATA_DIR = EXPERIMENT_DIR / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

THIS_EXPERIMENT_DIR = EXPERIMENT_DIR / EXPERIMENT_NUMBER
MODEL_DIR = THIS_EXPERIMENT_DIR / "model"
TEXT_VECTORIZER_PATH = MODEL_DIR / "text_vectorizer.pkl"

PREPROCESSED_DATA_DIR = THIS_EXPERIMENT_DIR / "preprocessed"
TRAIN_PREPROCESSED_PATH = PREPROCESSED_DATA_DIR / "train.pkl"
VAL_PREPROCESSED_PATH = PREPROCESSED_DATA_DIR / "val.pkl"

if not PREPROCESSED_DATA_DIR.exists():
    PREPROCESSED_DATA_DIR.mkdir(parents=True)
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True)
