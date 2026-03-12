import cv2
import mediapipe as mp
import numpy as np
import joblib

from predictor import predict
from display import draw_hand

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.9)

    model = joblib.load('../models/svc.pkl')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Impossible d'ouvrir la caméra.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossible d'ouvrir la caméra.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                predicted_label, proba = predict(model, points)
                draw_hand(frame, points, predicted_label, proba)
                
        cv2.imshow('Prédiction ASL', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()