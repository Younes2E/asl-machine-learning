import cv2
import mediapipe as mp
import numpy as np
import os
import joblib

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

labels = [chr(i) for i in range(97, 123)]

model = joblib.load('../models/asl_model.pkl')

def distance(X1, X2):
    return np.linalg.norm(X1-X2)

def predict(model, points):
    X = points.copy()
    dist_norm = distance(points[0],points[9])
    middle = np.mean(X, axis=0)
    X -= middle
    X /= dist_norm
    prediction = model.predict([X.reshape(42,)])

    return labels[int(prediction[0])], np.max(model.predict_proba([X.reshape(42,)]))


def draw_info_hand(frame, points, predicted_label, proba, hand_landmarks=None):
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    h, w, _ = frame.shape
    x_min, y_min = int(x_min * w), int(y_min * h)
    x_max, y_max = int(x_max * w), int(y_max * h)
    r_rect = np.max([x_max-x_min, y_max-y_min]) + 30
    x_rect = (x_min+x_max - r_rect)//2
    y_rect = (y_min+y_max - r_rect)//2

    #cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (0, 255, 0), 2)

    sub_img = frame[np.max([0,y_rect]):np.min([h,y_rect+r_rect]), np.max([0,x_rect]):np.min([w,x_rect+r_rect])]
    rect_alpha = np.zeros(sub_img.shape, dtype=np.uint8)
    alpha = 0.1
    res = cv2.addWeighted(sub_img, 1-alpha, rect_alpha, alpha, 1.0)
    frame[np.max([0,y_rect]):np.min([h,y_rect+r_rect]), np.max([0,x_rect]):np.min([w,x_rect+r_rect])] = res

    color = (32, 255, 32)

    #cv2.rectangle(frame, (x_rect, y_rect), (x_rect+r_rect, y_rect+r_rect), color, 3)

    text_lettre = f"{predicted_label}"
    (w_text, h_text) , _ = cv2.getTextSize(text_lettre, cv2.FONT_HERSHEY_DUPLEX, 3,3)
    x_text = (x_min+x_max)//2 - w_text//2
    y_text = (y_min+y_max)//2 + h_text//2
    cv2.putText(frame, text_lettre,(x_text,y_text) , cv2.FONT_HERSHEY_DUPLEX, 3, color, 3)

    text_proba =  f"{int(np.round(proba*100))}%" 
    (w_text_proba, h_text_proba) , _ = cv2.getTextSize(text_proba, cv2.FONT_HERSHEY_DUPLEX, 1,3)
    x_text_proba = (x_min+x_max)//2 - w_text_proba//2
    y_text_proba = (y_min+y_max)//2 + h_text_proba//2 + h_text + 5
    cv2.putText(frame, text_proba,(x_text_proba,y_text_proba) , cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)





cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur : Impossible de lire la caméra.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_classification in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness = hand_classification.classification[0].label
            points = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            predicted_label, proba = predict(model, points)
            draw_info_hand(frame, points, predicted_label, proba, hand_landmarks=None)
            

    cv2.imshow('Détection ASL', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
