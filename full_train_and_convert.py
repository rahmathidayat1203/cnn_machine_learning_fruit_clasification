import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Fungsi untuk mengkonversi model ke TFLite
def convert_to_tflite(model, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Model TFLite saved to {filename}")

# Fungsi untuk mengkonversi TFLite ke array C
def convert_tflite_to_c_array(tflite_model_path, c_array_path):
    with open(tflite_model_path, 'rb') as file:
        tflite_model = file.read()
    
    c_array = '{'
    c_array += ', '.join([f'0x{byte:02x}' for byte in tflite_model])
    c_array += '}'
    
    with open(c_array_path, 'w') as file:
        file.write(f"const unsigned char model_data[] = {c_array};\n")
        file.write(f"const unsigned int model_data_len = {len(tflite_model)};\n")
    
    print(f"C array saved to {c_array_path}")

# Kode Anda yang ada
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

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test accuracy: {test_acc}')

# Menyimpan model
model.save("model/model_buah.h5")
print("Model saved as H5")

# Mengkonversi ke TFLite
convert_to_tflite(model, "model/model_buah.tflite")

# Mengkonversi TFLite ke array C
convert_tflite_to_c_array("model/model_buah.tflite", "model/model_buah.c")