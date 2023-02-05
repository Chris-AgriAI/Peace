import cv2
import numpy as np
import tensorflow as tf

# Load the model onto the Edge TPU
interpreter = tf.lite.Interpreter(model_path="efficientdet-lite-peace_edgetpu.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    # Preprocess the image for the model
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data / 255.0).astype(np.float32)
    
    # Run the model on the image
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Check if "peace" label is present in the output
    if np.argmax(output_data) == 0:
        label = "peace"
    else:
        label = "not peace"
    
    # Display the result on the image
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the image
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
