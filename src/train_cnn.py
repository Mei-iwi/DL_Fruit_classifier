from config import MODEL_DIR, MODEL_PATH
from model_cnn import main_test_fruit_model_build

def main_test_train_cnn():
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # Viết train tại đây và lưu model

    #history = model.fit()
    #model.save(MODEL_PATH)

    print("[OK] train_cnn placeholder")


if __name__ == "__main__":
    main_test_train_cnn()
