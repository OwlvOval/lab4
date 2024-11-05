# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:10:27 2024

@author: Owl
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузка изображения и перевод его в оттенки серого
image = cv2.imread('chi.jpg')  # Укажите путь к вашему изображению
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Пороговая фильтрация для выделения текста
_, binary = cv2.threshold(gray,120, 255, cv2.THRESH_BINARY_INV)

# Морфологическая операция для удаления горизонтальных линий
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# Удаление линий из бинарного изображения
text_without_lines = cv2.subtract(binary, detected_lines)

# Поиск контуров
contours, _ = cv2.findContours(text_without_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Создаем пустое изображение для восстановления контуров текста
restored_image = np.ones_like(gray) * 255  # Белый фон

# Отрисовываем контуры на новом изображении
cv2.drawContours(restored_image, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

# Сохраняем результат как отдельное изображение
cv2.imwrite('restored_text_image.png', restored_image)
print("Обработанное изображение сохранено как 'restored_text_image.png'")

# Отображаем исходное и обработанное изображение для сравнения
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Исходное изображение")
plt.imshow(gray, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Изображение после удаления линий и восстановления текста")
plt.imshow(restored_image, cmap='gray')
plt.show()

