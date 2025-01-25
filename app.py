import cv2
import pickle
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Emotion Detection App",  # Title shown on the browser tab
    page_icon="✨",    # Emoji or icon for the app
    initial_sidebar_state="collapsed"   # Default state of the sidebar
)     

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Real-Time Face Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Capture an image and let AI detect your emotions in real time!</p>", unsafe_allow_html=True)

# Load the pre-trained emotion detection model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels for the 7 emotions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


# Add an emoji mapping for emotions
emoji_map = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

pic = st.camera_input("Take a picture")

if pic is not None:
    img = Image.open(pic)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    st.write(f"Number of faces detected: {len(faces)}")

    if len(faces) == 0:
        st.warning("No faces detected. Please try again.")
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            
            face_features = extract_features(face)
            prediction = model.predict(face_features)

            emotion = labels[np.argmax(prediction)]
             # Add Emotion Tag
            st.markdown(f"""
                <div style="background-color: #FFD700; padding: 10px; border-radius: 10px; text-align: center; font-size: 24px;">
                    <strong>{emotion}</strong> {emoji_map.get(emotion, '')}
                </div>
                """, unsafe_allow_html=True)

            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            img = cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Processed Image with Emotion Detection", use_container_width=True)
