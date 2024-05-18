import cv2
import mediapipe as mp
import numpy as np
import math
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Functions to calculate angles between three points
def calc_angle(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    # Calculating cosine of the angle between vectors
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle)
    
    # To make straight finger 0 degrees
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    
    return angle_deg

# initializing camera
def camera_req():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        cap.release()
        return None
    cap.release()
    print("Webcam permission granted and working correctly.")
    return cap
cap = camera_req()
if cap is None:
    exit("Webcam is not available. Exiting...")
cap = cv2.VideoCapture(0)

# Setting up mediapipe
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Drawing the hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # landmark coordinates
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                
                finger_angles = {}
                finger_angles['Thumb MCP'] = calc_angle(landmarks[2], landmarks[1], landmarks[0])
                finger_angles['Thumb IP'] = calc_angle(landmarks[3], landmarks[2], landmarks[1])
                finger_angles['Index MCP'] = calc_angle(landmarks[5], landmarks[6], landmarks[0])
                finger_angles['Index PIP'] = calc_angle(landmarks[6], landmarks[7], landmarks[8])
                finger_angles['Index DIP'] = calc_angle(landmarks[7], landmarks[8], landmarks[9])
                finger_angles['Middle MCP'] = calc_angle(landmarks[9], landmarks[10], landmarks[0])
                finger_angles['Middle PIP'] = calc_angle(landmarks[10], landmarks[11], landmarks[12])
                finger_angles['Middle DIP'] = calc_angle(landmarks[11], landmarks[12], landmarks[13])
                finger_angles['Ring MCP'] = calc_angle(landmarks[13], landmarks[14], landmarks[0])
                finger_angles['Ring PIP'] = calc_angle(landmarks[14], landmarks[15], landmarks[16])
                finger_angles['Ring DIP'] = calc_angle(landmarks[15], landmarks[16], landmarks[17])              
                finger_angles['Pinky MCP'] = calc_angle(landmarks[17], landmarks[18], landmarks[0])
                finger_angles['Pinky PIP'] = calc_angle(landmarks[18], landmarks[19], landmarks[20])
                finger_angles['Pinky DIP'] = calc_angle(landmarks[19], landmarks[20], landmarks[0])  

                wrist_angle = calc_angle(landmarks[0], landmarks[9], landmarks[17])

                for idx, (joint, angle) in enumerate(finger_angles.items()):
                    cv2.putText(frame, f'{joint}: {int(angle)}°', 
                                (10, 30 + idx * 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 0, 0), 
                                1, 
                                cv2.LINE_AA)
                
                cv2.putText(frame, f'Wrist: {int(wrist_angle)}°', 
                            (10, 30 + len(finger_angles) * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 0, 0), 
                            1, 
                            cv2.LINE_AA)
                
        cv2.imshow('Hand Tracking', frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
