import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['legsraise', 'dumbbell', 'babel']         #행동 변수
seq_length = 21                                  #윈도우 사이즈
created_time = int(time.time())                 #행동 녹화 시간

# Mediapipe hands model initialize
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity = 1,
    min_detection_confidence = 0.5, #ss
    min_tracking_confidence = 0.5)  #

path_dir = 'C://project/dataset/sourceData/1/C'
folder_list = os.listdir(path_dir)

os.makedirs('created_dataset', exist_ok=True)

   
for idx, action in enumerate(actions):
    data = []
    action_dir = path_dir + '/' + folder_list[idx]
      
    for frame in os.listdir(action_dir):
        
        img = 'C://project/dataset/sourceData/1/C/033-1-1-21-Z17_C/033-1-1-21-Z17_C-0000007.jpg'
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)          # 프레임을 읽어 Mediapipe에 넣어준다.

            # 좌표 설정
        if result.pose_landmarks is not None:
            joint = np.zeros((33,4))
            for j,lm in enumerate(result.pose_landmarks.landmark):
                joint[j] = [lm.x, lm.y,lm.z,lm.visibility]
                
                    
            stand_joint = (joint[12] + joint[11] + joint[24] + joint[23])/4    
            locVector = joint[:, :3] - stand_joint[:3]
            locVector = np.concatenate([locVector,joint[:,[3]]], axis = 1)
            
            
            """
            
            angle = np.arccos(np.einsum('nt, nt->n',
                v[[0,0,3,3,1,4,6,7],:],
                v[[1,2,4,5,6,7,8,9],:]))
            angle = np.degrees(angle) / 360.0
                    
            angle_label = np.array([angle], dtype = np.float32)     # Lable
            angle_label = np.append(angle_label, idx)
                      
            
            d = np.concatenate([joint.flatten(), angle_label]) # 추가해주기

            data.append(d)

            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        """
        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break
print(joint[11])
print(stand_joint)
print(locVector[11])
print(idx)
"""
    data = np.array(data)
    print(action, data.shape)
    np.save(os.path.join('created_dataset', f'raw_{action}_{created_time}'),data)

    full_seq_data=[]
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])
    
    
    full_seq_data = np.array(full_seq_data)
    print(action, full_seq_data.shape)
    np.save(os.path.join('created_dataset', f'seq_{action}_{created_time}'), full_seq_data)
    """