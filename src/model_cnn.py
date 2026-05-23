import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape, num_classes, learning_rate=0.001):
    """
    Xây dựng mô hình CNN cơ bản, không dùng model huấn luyện sẵn.
    """

    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=input_shape),

        layers.Conv2D(16, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model