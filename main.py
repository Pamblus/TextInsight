import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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
    X = np.array(X)
    y = np.array(y)
    return X, y

# Функция для создания и обучения модели
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Функция для предсказания
def predict(model, word):
    word_sequence = np.array([list(map(ord, word))])
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
            model = train_model(X, y)
            print("Обучение завершено.")
        
        elif choice == '2':
            word = input("Введите слово для предсказания: ")
            prediction = predict(model, word)
            print("Предсказанные числа: ", ', '.join(map(str, prediction.astype(int))))
        
        elif choice == '3':
            break
        
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()
