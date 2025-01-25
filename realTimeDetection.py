import cv2
import pickle
import numpy as np

import sys
import io

# Set the standard output and error encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load the pre-trained emotion detection model from pickle file
with open('model.pkl', 'rb') as file:  # Replace 'model.pkl' with the correct pickle file name
    model = pickle.load(file)

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels for the 7 emotions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to preprocess the face image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshaping for model input
    return feature / 255.0  # Normalize the image

# Start capturing video from the webcam
webcam = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not webcam.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Set resolution (optional)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read a frame from the webcam
    ret, im = webcam.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            # Extract the face region from the image
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)  # Draw rectangle around face

            # Resize face to match input size expected by the model
            image = cv2.resize(image, (48, 48))

            # Preprocess the face for prediction
            img = extract_features(image)

            # Predict the emotion
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]  # Get the emotion with the highest probability

            # Convert prediction_label to a byte string to handle non-ASCII characters
            prediction_label = prediction_label.encode('utf-8', errors='ignore').decode('utf-8')

            # Display the emotion label on the video feed
            cv2.putText(im, prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the video frame with face detection and emotion prediction
        cv2.imshow("Emotion Detection", im)

        # Break the loop on pressing 'Esc' key
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the Esc key
            break
    except Exception as e:
        print(f"Error during prediction: {e}")
        pass

# Release the webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
