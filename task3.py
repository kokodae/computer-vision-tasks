import tensorflow as tf 
from tensorflow.keras import layers, models 
import matplotlib.pyplot as plt 
(train_images, train_labels), (test_images, test_labels) = 
tf.keras.datasets.cifar10.load_data() 
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['Самолет', 'Автомобиль', 'Птица', 'Кошка', 'Олень', 'Собака', 
'Лягушка', 'Лошадь', 'Корабль', 'Грузовик'] 

model = models.Sequential([ 
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), 
layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.Flatten(), 
    layers.Dense(64, activation='relu'), 
    layers.Dense(10) 
]) 
 
model.summary() 

model.compile(optimizer='adam', 
              
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy']) 
 
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels)) 
 
plt.plot(history.history['loss'], label='Функция потерь на обучающем 
наборе') 
plt.plot(history.history['val_loss'], label='Функция потерь на валидационном 
наборе') 
plt.xlabel('Эпоха') 
plt.ylabel('Функция потерь') 
plt.legend() 
plt.show() 

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) 
print(f'Точность на тестовом наборе данных: {test_acc}') 
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) 
 
predictions = probability_model.predict(test_images) 
 
for i in range(5): 
    plt.figure(figsize=(2, 2)) 
    plt.imshow(test_images[i]) 
    plt.title(f'Предсказано: {class_names[tf.argmax(predictions[i])]}, ' 
              f'Действительное: {class_names[test_labels[i][0]]}') 
    plt.axis('off') 
    plt.show()
