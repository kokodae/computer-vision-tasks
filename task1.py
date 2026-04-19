import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from google.colab.patches import cv2_imshow 

image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE) 
 
# Оператор Робертса 
roberts_x = np.array([[1, 0], [0, -1]]) 
roberts_y = np.array([[0, 1], [-1, 0]]) 
roberts_x_image = cv2.filter2D(image, -1, roberts_x) 
roberts_y_image = cv2.filter2D(image, -1, roberts_y) 
roberts_image = np.sqrt(roberts_x_image**2 + roberts_y_image**2) 
 
# Оператор Превитта 
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) 
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) 
prewitt_x_image = cv2.filter2D(image, -1, prewitt_x) 
prewitt_y_image = cv2.filter2D(image, -1, prewitt_y) 
prewitt_image = np.sqrt(prewitt_x_image**2 + prewitt_y_image**2) 
 
# Оператор Собеля 
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) 
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) 
sobel_image = np.sqrt(sobel_x**2 + sobel_y**2) 

plt.figure(figsize=(15, 10)) 
plt.subplot(2, 2, 1) 
plt.title('Оригинальное изображение') 
plt.imshow(image, cmap='gray') 
 
plt.subplot(2, 2, 2) 
plt.title('Оператор Робертса') 
plt.imshow(roberts_image, cmap='gray') 
 
plt.subplot(2, 2, 3) 
plt.title('Оператор Превитта') 
plt.imshow(prewitt_image, cmap='gray') 
 
plt.subplot(2, 2, 4) 
plt.title('Оператор Собеля') 
plt.imshow(sobel_image, cmap='gray') 
 
plt.show()
