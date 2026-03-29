import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Change this label for each sign
label = "I"

data_file = "landmark_data.csv"

cap = cv2.VideoCapture(0)

with open(data_file, mode='a', newline='') as f:
    writer = csv.writer(f)

    print("Press 'S' to save landmark")
    print("Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Extract 63 values (21 landmarks × 3)
                row = []
                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)
                    row.append(lm.z)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    row.append(label)
                    writer.writerow(row)
                    print("Saved:", label)

                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        cv2.imshow("Collect Landmarks", frame)