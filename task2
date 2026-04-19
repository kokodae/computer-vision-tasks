import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

train_data_dir = '/content/fruits_dataset/fruits-360_dataset_100x100/fruits-360/Training' 
test_data_dir = '/content/fruits_dataset/fruits-360_dataset_100x100/fruits-360/Test' 

train_datagen = ImageDataGenerator( 
    rescale=1./255, 
    validation_split=0.2 
) 
 
train_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(100, 100), 
    batch_size=32, 
    class_mode='categorical', 
    subset='training' 
) 
 
validation_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(100, 100), 
    batch_size=32, 
    class_mode='categorical', 
    subset='validation' 
) 
 
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory( 
    test_data_dir, 
    target_size=(100, 100), 
    batch_size=32, 
    class_mode='categorical' 
) 
print(train_generator.class_indices) 
 
 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
 
model = Sequential([ 
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)), 
    MaxPooling2D(pool_size=(2, 2)), 
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D(pool_size=(2, 2)), 
    Conv2D(128, (3, 3), activation='relu'), 
    MaxPooling2D(pool_size=(2, 2)), 
    Flatten(), 
    Dense(512, activation='relu'), 
    Dropout(0.5), 
    Dense(train_generator.num_classes, activation='softmax') 
]) 
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
 
history = model.fit( 
    train_generator, 
    steps_per_epoch=train_generator.samples // train_generator.batch_size, 
    validation_data=validation_generator, 
    validation_steps=validation_generator.samples // validation_generator.batch_size, 
    epochs=10)

model.save('fruit_classifier.keras') 
 
from tensorflow.keras.models import load_model 
model = load_model('fruit_classifier.keras') 

def plot_images_with_predictions(generator, model, num_images=5): 
    class_labels = list(generator.class_indices.keys()) 
    for i in range(num_images): 
        images, labels = next(generator) 
        # Выбираем случайное изображение из батча 
        idx = np.random.randint(0, len(images)) 
        img = images[idx] 
        true_label = class_labels[np.argmax(labels[idx])] 
        
        img_array = np.expand_dims(img, axis=0) 
        predictions = model.predict(img_array) 
        predicted_label = class_labels[np.argmax(predictions)] 
        
        plt.figure() 
        plt.imshow(img) 
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}") 
        plt.axis('off') 
    plt.show() 

plot_images_with_predictions(test_generator, model, num_images=5) 
