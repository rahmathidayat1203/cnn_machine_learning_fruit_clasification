import tensorflow as tf

# Konversi SavedModel ke TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model/model_buah")
tflite_model = converter.convert()

# Simpan model TFLite
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)