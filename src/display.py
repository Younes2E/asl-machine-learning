import cv2
import numpy as np

def draw_hand(frame, points, predicted_label, proba):
    x_min, y_min = np.min(points, axis=0)[:2]
    x_max, y_max = np.max(points, axis=0)[:2]

    h, w, _ = frame.shape
    x_min, y_min = int(x_min * w), int(y_min * h)
    x_max, y_max = int(x_max * w), int(y_max * h)

    r_rect = np.max([x_max-x_min, y_max-y_min]) + 30
    x_rect = (x_min+x_max - r_rect)//2
    y_rect = (y_min+y_max - r_rect)//2

    sub_img = frame[np.max([0,y_rect]):np.min([h,y_rect+r_rect]), np.max([0,x_rect]):np.min([w,x_rect+r_rect])]
    rect_alpha = np.zeros(sub_img.shape, dtype=np.uint8)
    alpha = 0.1
    res = cv2.addWeighted(sub_img, 1-alpha, rect_alpha, alpha, 1.0)
    frame[np.max([0,y_rect]):np.min([h,y_rect+r_rect]), np.max([0,x_rect]):np.min([w,x_rect+r_rect])] = res

    color = (32, 255, 32)

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

