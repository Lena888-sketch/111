import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Загружаем данные (MNIST: цифры 0-9)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Нормализуем пиксели в диапазон [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Добавляем канал (для совместимости с Conv2D)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 4. Строим простую CNN-модель
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")  # 10 классов: цифры 0..9
])

# 5. Компилируем модель
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Обучаем
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 7. Проверяем на тестовой выборке
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Точность на тесте: {test_acc:.4f}")

# 8. Пример предсказания для первых 5 изображений
import numpy as np
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)
print("Предсказанные цифры:", predicted_labels)
print("Реальные цифры:", y_test[:5])
