# mengimpor library
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# menggunakan ImageDataGenerator dari Keras untuk memuat dan memproses gambar dari direktori

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# memuat data dari direktori

train_generator = train_datagen.flow_from_directory(
    'fruits/Training',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'fruits/Test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Membuat model CNN
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(train_generator.num_classes, activation='softmax')
])
# Kompilasi model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Evaluasi model
test_loss, test_acc = model.evaluate(validation_generator)

# Setelah melatih model
tf.saved_model.save(model, "model/model_buah")
print(f'Test accuracy: {test_acc}')