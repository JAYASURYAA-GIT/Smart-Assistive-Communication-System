import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# =========================
# Word Builder Variables
# =========================

current_word = ""
previous_letter = ""
last_added_letter = ""
frame_count = 0

# =========================
# Load Model
# =========================

model = load_model("models/landmark_model.h5")

# =========================
# Load Labels
# =========================

data = pd.read_csv("data/landmark_data.csv", header=None)
labels = data.iloc[:, -1].values

encoder = LabelEncoder()
encoder.fit(labels)

# =========================
# Initialize MediaPipe
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# =========================
# Start Camera
# =========================

cap = cv2.VideoCapture(0)

# =========================
# Real-Time Detection Loop
# =========================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # =========================
            # Extract 63 Features
            # =========================

            row = []

            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row = np.array(row).reshape(1, -1)

            # =========================
            # Model Prediction
            # =========================

            prediction = model.predict(row, verbose=0)

            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            letter = encoder.inverse_transform([predicted_class])[0]

            # =========================
            # Word Builder Logic
            # =========================

            if letter == previous_letter:
                frame_count += 1
            else:
                previous_letter = letter
                frame_count = 1

            # if same letter seen for enough frames
            if frame_count == 15:
                current_word += letter
                print("Word:", current_word)

            # =========================
            # Display Prediction
            # =========================

            cv2.putText(frame,
                        f"Prediction: {letter} ({confidence:.2f})",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

    # =========================
    # Display Word
    # =========================

    cv2.putText(frame,
                f"Word: {current_word}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.imshow("Sign Translator", frame)

    # =========================
    # Keyboard Controls
    # =========================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        current_word = ""
        previous_letter = ""
        frame_count = 0

    if key == ord('q'):
        break

# =========================
# Release Camera
# =========================

cap.release()
cv2.destroyAllWindows()