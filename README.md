# Задачи со сверточными нейронными сетями

## Задача 1: Детектирование границ

Реализация операторов Робертса, Превитта и Собеля для выделения границ на изображении.

**Файл:** [task1.py](https://github.com/kokodae/computer-vision-tasks/blob/main/task1.py)

**Результат:** result1.PNG

**Зависимости:** opencv-python, numpy, matplotlib

---

## Задача 2: Классификация фруктов

CNN для классификации датасета Fruits-360 (100x100, ~131 класс). Архитектура: Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(128) -> MaxPool -> Flatten -> Dense(512) -> Dropout(0.5) -> Dense(num_classes).

**Файл:** [task2.PY](https://github.com/kokodae/computer-vision-tasks/blob/main/task2.py)

**Результат:** result2.PNG

---

## Задача 3: Классификация CIFAR-10

CNN для датасета CIFAR-10 (32x32, 10 классов: самолет, автомобиль, птица, кошка, олень, собака, лягушка, лошадь, корабль, грузовик). Архитектура: Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(64) -> Flatten -> Dense(64) -> Dense(10).

**Файл:** [task3.PY](https://github.com/kokodae/computer-vision-tasks/blob/main/task3.py)

**Результат:** result3.PNG

---

## Установка

pip install opencv-python numpy matplotlib tensorflow

---

## Примечания

Задача 1 требует файл 2.jpg в директории. Задача 2 требует скачать датасет Fruits-360. Задача 3 загружает CIFAR-10 автоматически.
