import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mostrar apenas erros

import cv2
import mediapipe as mp
import numpy as np
from services.analysis_services import calculate_angle, determine_phase, get_phase_color, draw_landmarks, draw_angle, calculate_velocity, calculate_acceleration

def process_video(video_path: str, output_path: str):
    
    # Initializing pose and drawing utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initializing video capture
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # Initializing 'previous' variables to calculate movement
    prev_landmarks = None
    prev_time = None
    prev_velocities = None
    
    while cap.isOpened():
        
        # Initializing frame capture
        ret, frame = cap.read()
        if not ret:
            break

        # Getting current time
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # current time in seconds

        # Getting image from frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Getting pose from image
        results = pose.process(image)

        if results.pose_landmarks:
            
            # Getting landmarks from pose
            landmarks = results.pose_landmarks.landmark
            
            # Getting body landmarks (connections) from landmarks
            body_landmarks = {i: landmarks[i] for i in range(len(landmarks)) if i not in range(0, 11)}
            
            # Comparing shoulder and ankle direction for algorithm calibration
            # shoulder.x and anlke.x
            if landmarks[12].x > landmarks[28].x:
                cv2.putText(frame, '-0', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (1,1,1), 2, cv2.LINE_AA)
            
            if landmarks[12].x < landmarks[28].x:
                cv2.putText(frame, '0-', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (1,1,1), 2, cv2.LINE_AA)

            # Reference angles object
            reference_angles = {
                'shoulder':{
                    
                    'right': {
                        'angle': calculate_angle(landmarks[14], landmarks[12], landmarks[24]),
                        'visibility': landmarks[12].visibility,
                        'node': landmarks[12]
                    },
                    'left': {
                        'angle': calculate_angle(landmarks[13], landmarks[11], landmarks[23]),
                        'visibility': landmarks[11].visibility,
                        'node': landmarks[11]
                    },
                },
                
                'hip':{
                    
                    'right': {
                        'angle': calculate_angle(landmarks[12], landmarks[24], landmarks[25]),
                        'visibility': landmarks[24].visibility,
                        'node': landmarks[24]
                    },
                    'left': {
                        'angle': calculate_angle(landmarks[11], landmarks[23], landmarks[26]),
                        'visibility': landmarks[23].visibility,
                        'node': landmarks[23]
                    },
                },
                
                'elbow': {
                    
                    'right': {
                        'angle': calculate_angle(landmarks[14], landmarks[12], landmarks[16]),
                        'visibility': landmarks[14].visibility,
                        'node': landmarks[14]
                    },
                    'left': {
                        'angle': calculate_angle(landmarks[13], landmarks[11], landmarks[15]),
                        'visibility': landmarks[13].visibility,
                        'node': landmarks[13]
                    },
                },
                
                'knee': {
                    
                    'right': {
                        'angle': calculate_angle(landmarks[24], landmarks[26], landmarks[28]),
                        'visibility': landmarks[26].visibility,
                        'node': landmarks[26]
                    },
                    'left': {
                        'angle': calculate_angle(landmarks[23], landmarks[25], landmarks[27]),
                        'visibility': landmarks[25].visibility,
                        'node': landmarks[25]
                    }
                },
            }
            
            # Getting phase from angles and landmarks
            phase = determine_phase(reference_angles, landmarks)
            
            # Getting color corresponding to phase
            color = get_phase_color(phase)
            
            # Display phase text in screen and drawing landmarks
            cv2.putText(frame, phase, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            draw_landmarks(frame, body_landmarks, mp_pose.POSE_CONNECTIONS, color)

            # Draw angles
            # draw_angle(frame, landmarks[24], landmarks[12], landmarks[26], reference_angles['RIGHT_SHOULDER']['angle'], color)
            # draw_angle(frame, landmarks[23], landmarks[11], landmarks[25], reference_angles['LEFT_SHOULDER']['angle'], color)
            # draw_angle(frame, landmarks[12], landmarks[24], landmarks[26], reference_angles['RIGHT_HIP']['angle'], color)
            # draw_angle(frame, landmarks[11], landmarks[23], landmarks[25], reference_angles['LEFT_HIP']['angle'], color)
            # draw_angle(frame, landmarks[24], landmarks[26], landmarks[28], reference_angles['RIGHT_KNEE']['angle'], color)
            # draw_angle(frame, landmarks[23], landmarks[25], landmarks[27], reference_angles['LEFT_KNEE']['angle'], color)
            # draw_angle(frame, landmarks[11], landmarks[13], landmarks[15], reference_angles['LEFT_ELBOW']['angle'], color)
            # draw_angle(frame, landmarks[12], landmarks[14], landmarks[16], reference_angles['RIGHT_ELBOW']['angle'], color)

            # Calculating movement
            if prev_landmarks is not None and prev_time is not None:
                
                time_interval = current_time - prev_time
                velocities = {}
                accelerations = {}

                # Getting point_name and landmark_index from reference angles
                for point_name, landmark_index in [('RIGHT_ELBOW', 14), ('LEFT_ELBOW', 13),
                                                   ('RIGHT_KNEE', 26), ('LEFT_KNEE', 25),
                                                   ('RIGHT_HIP', 24), ('LEFT_HIP', 23),
                                                   ('RIGHT_SHOULDER', 12), ('LEFT_SHOULDER', 11)]:
                    
                    # Getting previous and current position from landmarks dictionary
                    prev_pos = [prev_landmarks[landmark_index].x, prev_landmarks[landmark_index].y, prev_landmarks[landmark_index].z]
                    curr_pos = [landmarks[landmark_index].x, landmarks[landmark_index].y, landmarks[landmark_index].z]

                    # Calculating velocity and storing in velocities dict
                    velocity = calculate_velocity(prev_pos, curr_pos, time_interval)
                    velocities[point_name] = velocity

                    # Display velocities in screen
                    # cv2.putText(frame, f'V_{point_name}: {np.linalg.norm(velocity):.2f}', (int(curr_pos[0] * frame.shape[1]), int(curr_pos[1] * frame.shape[0]) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

                    if prev_velocities is not None:
                        
                        # Calculating acceleration and storing to accelerations dict
                        acceleration = calculate_acceleration(prev_velocities[point_name], velocity, time_interval)
                        accelerations[point_name] = acceleration

                        # Display acceleration in screen
                        # cv2.putText(frame, f'A_{point_name}: {np.linalg.norm(acceleration):.2f}', (int(curr_pos[0] * frame.shape[1]),
                        #                                                                            int(curr_pos[1] * frame.shape[0]) - 20), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
                        #             cv2.LINE_AA)

                # Storing current velocity in prev_vellocity for next loop iteration
                prev_velocities = velocities

            # Doing the same for landmarks and time
            prev_landmarks = landmarks
            prev_time = current_time

            print("Angles:", reference_angles)
            print("Phase:", phase)

        out.write(frame)

    cap.release()
    out.release()