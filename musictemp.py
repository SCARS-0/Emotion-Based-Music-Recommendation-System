import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Load the pre-trained model and label
model = load_model("C:/Users/Deshmukh/Desktop/liveEmoji-main/model.h5")
label = np.load("C:/Users/Deshmukh/Desktop/liveEmoji-main/labels.npy")

# Initialize MediaPipe holistic and drawing utilities
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


# Function to process a frame and predict emotion
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holis.process(frame_rgb)

    landmarks = []

    if result.face_landmarks:
        for i in result.face_landmarks.landmark:
            landmarks.append(i.x - result.face_landmarks.landmark[1].x)
            landmarks.append(i.y - result.face_landmarks.landmark[1].y)

        if result.left_hand_landmarks:
            for i in result.left_hand_landmarks.landmark:
                landmarks.append(i.x - result.left_hand_landmarks.landmark[8].x)
                landmarks.append(i.y - result.left_hand_landmarks.landmark[8].y)
        else:
            landmarks.extend([0.0] * 42)

        if result.right_hand_landmarks:
            for i in result.right_hand_landmarks.landmark:
                landmarks.append(i.x - result.right_hand_landmarks.landmark[8].x)
                landmarks.append(i.y - result.right_hand_landmarks.landmark[8].y)
        else:
            landmarks.extend([0.0] * 42)

        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = label[np.argmax(model.predict(landmarks))]

        return prediction
    return None


# OpenCV video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

emotion = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    detected_emotion = process_frame(frame)

    if detected_emotion:
        emotion = detected_emotion
        cv2.putText(frame, emotion, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the detected emotion
print(f"Detected Emotion: {emotion}")
