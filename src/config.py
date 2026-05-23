from pathlib import Path

# =========================
# PROJECT PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
SOURCE_TEST_DIR = DATA_DIR / "test_original"

PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

TEST_IMAGES_DIR = DATA_DIR / "test_images"

REPORT_DIR = PROJECT_ROOT / "reports" / "01_data_profile"

DATA_PROFILE_PATH = REPORT_DIR / "data_profile.csv"
BAD_IMAGES_PATH = REPORT_DIR / "bad_images.csv"
CLASS_DISTRIBUTION_PATH = REPORT_DIR / "class_distribution.png"
DATASET_README_PATH = REPORT_DIR / "dataset_readme.md"

DATA_SUMMARY_PATH = DATA_DIR / "data_summary.csv"

# =========================
# DATASET CONFIG
# =========================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

VAL_RATIO = 0.2
TEST_RATIO = 0.2

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

MIN_IMAGES_PER_CLASS = 20

# Đường dẫn điểm lưu models

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# DATA
# =========================
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# =========================
# MODEL
# =========================
MODEL_DIR = BASE_DIR / "models"
MODEL_NAME = "fruit_cnn.keras"
MODEL_PATH = MODEL_DIR / MODEL_NAME

# =========================
# REPORT
# =========================
REPORT_DIR = BASE_DIR / "reports"
TRAINING_REPORT_DIR = REPORT_DIR / "02_training"
HISTORY_PATH = TRAINING_REPORT_DIR / "history.csv"
MODEL_SUMMARY_PATH = TRAINING_REPORT_DIR / "model_summary.txt"

# =========================
# TRAINING CONFIG
# =========================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
SEED = 42

# Tự chia dữ liệu từ data/raw
VALIDATION_SPLIT = 0.2