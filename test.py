import cv2
import mediapipe as mp
import numpy as np
import time, os
from tensorflow.keras.models import load_model

actions = ['legraise', 'dumbbell', 'babel']
seq_length = 21

model = load_model('models/model.h5')

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity = 0,
    min_detection_confidence = 0.5, #
    min_tracking_confidence = 0.5
)   #

cap = cv2.VideoCapture(0)

"""
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
"""

seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()
    """
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    """
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks is not None:
        joint = np.zeros((33, 4))
        res = result.pose_landmarks
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
        
        
        stand_joint = (joint[12] + joint[11] + joint[24] + joint[23])/4    
        locVector = joint[:, :3] - stand_joint[:3]
        locVector = np.concatenate([locVector,joint[:,[3]]], axis = 1)
        
        
        v1 = joint[[12,12,14,11,11,13,24,23,26,25], :3]       # Vector 생성
        v2 = joint[[14,24,16,13,23,15,26,25,28,27], :3]
        v = v2 - v1
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] #뉴클리디언 디스턴스 -> eigen vector 생성
            
            
        angle = np.arccos(np.einsum('nt, nt->n',
            v[[0,0,3,3,1,4,6,7],:],
            v[[1,2,4,5,6,7,8,9],:]))
        angle = np.degrees(angle) / 180 # Convert radian to degree

        d = np.concatenate([locVector.flatten(), angle])

        seq.append(d)

        mp_drawing.draw_landmarks(img, res, mp_pose.POSE_CONNECTIONS)

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        y_pred = model.predict(input_data).squeeze()

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        if conf < 0.9:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

        cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[12].x * img.shape[1]), int(res.landmark[12].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break