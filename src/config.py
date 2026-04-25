from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "test_images"
MODEL_PATH = PROJECT_ROOT / "models" / "fruit_cnn.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
LEARNING_RATE = 0.001
