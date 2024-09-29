import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Функция для загрузки данных из JSON файла
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Функция для подготовки данных
def prepare_data(data):
    X = []
    y = []
    for entry in data:
        numbers = entry['numbers']
        word = entry['word']
        X.append(list(map(ord, word)))
        y.append(numbers)
    X = pad_sequences(X, maxlen=48, padding='post')
    y = np.array(y)
    return X, y

# Функция для создания модели
def create_model(input_shape):
    model = Sequential()
    model.add(Embedding(input_dim=128, output_dim=64, input_length=input_shape[0]))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(24, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Функция для обучения модели
def train_model(model, X, y, epochs=50, batch_size=32):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('model.h5')

# Функция для предсказания
def predict(model, word):
    word_sequence = pad_sequences([list(map(ord, word))], maxlen=48, padding='post')
    prediction = model.predict(word_sequence)
    return prediction.flatten()

# Основное меню
def main():
    while True:
        print("1. Начать обучение")
        print("2. Проверка знаний")
        print("3. Выход")
        choice = input("Выберите действие: ")
        
        if choice == '1':
            data = load_data('learn.json')
            X, y = prepare_data(data)
            model = create_model(X.shape[1:])
            train_model(model, X, y)
            print("Обучение завершено.")
        
        elif choice == '2':
            model = load_model('model.h5')
            word = input("Введите слово для предсказания: ")
            prediction = predict(model, word)
            print("Предсказанные числа: ", ', '.join(map(str, prediction.astype(int))))
        
        elif choice == '3':
            break
        
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()
