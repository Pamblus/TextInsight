import json
import random
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Функция для генерации случайного имени файла
def generate_filename():
    return f"training_data_{random.randint(10000000, 99999999)}.json"

# Функция для сохранения данных обучения
def save_training_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Функция для загрузки данных обучения
def load_training_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Функция для обработки данных
def process_data(input_data, vectorizer, classifier, answers):
    input_vector = vectorizer.transform([input_data])
    predicted_index = classifier.predict(input_vector)[0]
    return answers[predicted_index]

# Главное меню
def main_menu():
    while True:
        print("\nГлавное меню:")
        print("1. Ввести данные вручную")
        print("2. Обучение с данными")
        print("3. Показать файлы обучения")
        print("4. Удалить файлы обучения")
        print("5. Выход")
        choice = input("Выберите пункт меню: ")

        if choice == '1':
            manual_input()
        elif choice == '2':
            training_with_data()
        elif choice == '3':
            show_training_files()
        elif choice == '4':
            delete_training_files()
        elif choice == '5':
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

# Ручной ввод данных
def manual_input():
    input_data = input("Введите данные: ")
    correct_answer = input("Введите правильный ответ: ")
    training_data = {input_data: correct_answer}
    filename = generate_filename()
    save_training_data(training_data, filename)
    print(f"Данные сохранены в файл: {filename}")

# Обучение с данными
def training_with_data():
    print("Выберите файл для продолжения обучения или начните новое обучение.")
    files = [f for f in os.listdir() if f.startswith("training_data_")]
    if files:
        for i, file in enumerate(files):
            print(f"{i + 1}. {file}")
        file_choice = input("Введите номер файла или 'new' для нового обучения: ")
        if file_choice.isdigit() and 1 <= int(file_choice) <= len(files):
            filename = files[int(file_choice) - 1]
            training_data = load_training_data(filename)
        elif file_choice.lower() == 'new':
            training_data = {}
            filename = generate_filename()
        else:
            print("Неверный выбор.")
            return
    else:
        training_data = {}
        filename = generate_filename()

    iterations = int(input("Введите количество повторений: "))
    for _ in range(iterations):
        input_data = input("Введите данные: ")
        correct_answer = input("Введите правильный ответ: ")
        training_data[input_data] = correct_answer
        save_training_data(training_data, filename)
        print(f"Данные сохранены в файл: {filename}")

    # Преобразование данных для обучения
    input_texts = list(training_data.keys())
    answers = list(training_data.values())
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(input_texts)
    y = np.array(answers)

    # Обучение классификатора
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X, y)

    while True:
        input_data = input("Введите данные для обработки (или 'exit' для выхода): ")
        if input_data.lower() == 'exit':
            break
        result = process_data(input_data, vectorizer, classifier, answers)
        print(f"Результат: {result}")
        is_correct = input("Верный ответ? (да/нет): ").lower()
        if is_correct == 'нет':
            correct_answer = input("Введите правильный ответ: ")
            training_data[input_data] = correct_answer
            save_training_data(training_data, filename)
            print(f"Данные обновлены в файле: {filename}")

# Показать файлы обучения
def show_training_files():
    files = [f for f in os.listdir() if f.startswith("training_data_")]
    if files:
        for file in files:
            print(file)
    else:
        print("Файлы обучения отсутствуют.")

# Удалить файлы обучения
def delete_training_files():
    files = [f for f in os.listdir() if f.startswith("training_data_")]
    if files:
        for i, file in enumerate(files):
            print(f"{i + 1}. {file}")
        file_choice = input("Введите номер файла для удаления (или 'all' для удаления всех): ")
        if file_choice.isdigit() and 1 <= int(file_choice) <= len(files):
            os.remove(files[int(file_choice) - 1])
            print(f"Файл {files[int(file_choice) - 1]} удален.")
        elif file_choice.lower() == 'all':
            for file in files:
                os.remove(file)
            print("Все файлы обучения удалены.")
        else:
            print("Неверный выбор.")
    else:
        print("Файлы обучения отсутствуют.")

# Запуск программы
if __name__ == "__main__":
    main_menu()
