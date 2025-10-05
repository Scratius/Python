
import tensorflow as tf   # это библиотека для нейросетей
from tensorflow import keras  # из неё будем брать модель и данные
import matplotlib.pyplot as plt  # это чтобы показывать картинки

# Загружаем базу данных MNIST
# В ней уже есть 60 000 картинок цифр для обучения и 10 000 для проверки
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Данные изначально идут в виде чисел от 0 до 255 (яркость пикселя)
# Чтобы сеть работала лучше, нормализуем (делим на 255, чтобы получить от 0 до 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Создаём простую модель
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # превращаем 28х28 картинку в один длинный вектор
    keras.layers.Dense(128, activation='relu'),  # обычный слой с 128 нейронами, активация ReLU
    keras.layers.Dense(10, activation='softmax') # выходной слой, 10 цифр от 0 до 9
])

# Компилируем модель (говорим как обучать)
model.compile(
    optimizer='adam',                    # алгоритм обучения
    loss='sparse_categorical_crossentropy',  # функция ошибки для многоклассовой задачи
    metrics=['accuracy']                 # чтобы видеть точность
)

# Обучаем модель
# x_train - картинки, y_train - правильные ответы
model.fit(x_train, y_train, epochs=5)  # 5 эпох — то есть 5 проходов по всем данным

# Проверяем, как модель работает на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nТочность на тестовых данных:', test_acc)

# Берём случайную картинку из теста и смотрим предсказание
import random
n = random.randint(0, len(x_test) - 1)

plt.imshow(x_test[n], cmap='gray')  # показываем цифру
plt.title(f'Настоящая цифра: {y_test[n]}')
plt.show()

# Теперь попробуем узнать, что думает сеть
pred = model.predict(x_test[n].reshape(1, 28, 28))  # подаём одну картинку
print("Модель думает, что это цифра:", pred.argmax())  # argmax показывает номер максимального значения
