import cv2
import numpy as np
import tensorflow as tf  # Ganti tflite_runtime dengan tensorflow

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image, target_size):
    # Resize image
    image = cv2.resize(image, target_size)
    # Convert to RGB (OpenCV uses BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def load_model_and_predict(image, model_path, label_path):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    input_shape = input_details[0]['shape'][1:3]  # Exclude batch dimension
    processed_image = preprocess_image(image, input_shape)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Load labels
    labels = load_labels(label_path)

    # Get top prediction
    top_prediction = np.argmax(output_data[0])
    confidence = output_data[0][top_prediction]

    return labels[top_prediction], confidence

# Main loop for capturing from camera and running predictions
def predict_from_camera(model_path, label_path):
    # Open video capture (webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Perform prediction on the captured frame
        predicted_class, confidence = load_model_and_predict(frame, model_path, label_path)

        # Display prediction on the frame
        cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Camera Prediction', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Example usage
model_path = 'model/model_buah.tflite'
label_path = 'fruit_labels.txt'

predict_from_camera(model_path, label_path)
