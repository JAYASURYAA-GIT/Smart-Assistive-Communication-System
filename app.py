import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from flask import Flask, render_template, Response
import atexit
import time

app = Flask(__name__)

# ===============================
# LOAD MODEL AND LABELS
# ===============================

model = load_model("models/landmark_model.h5")

data = pd.read_csv("data/landmark_data.csv", header=None)
labels = data.iloc[:, -1].values

encoder = LabelEncoder()
encoder.fit(labels)

# ===============================
# MEDIAPIPE SETUP
# ===============================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

camera = cv2.VideoCapture(0)

# ===============================
# VIDEO STREAM FUNCTION
# ===============================

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        letter = "No Hand"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row = np.array(row).reshape(1, -1)

                if row.shape[1] == 63:
                    prediction = model.predict(row, verbose=0)
                    predicted_class = np.argmax(prediction)
                    letter = encoder.inverse_transform([predicted_class])[0]

        # Draw prediction text on video
        cv2.putText(frame,
                    f"Prediction: {letter}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)

# ===============================
# ROUTES
# ===============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===============================
# CLEAN SHUTDOWN
# ===============================

def shutdown():
    camera.release()

atexit.register(shutdown)

if __name__ == "__main__":
    app.run(debug=False)