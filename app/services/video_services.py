import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mostrar apenas erros

import cv2
import mediapipe as mp
import numpy as np
from services.analysis_services import calculate_angle, determine_phase, get_phase_color, draw_landmarks, draw_angle, calculate_velocity, calculate_acceleration

def process_video(video_path: str, output_path: str):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    prev_landmarks = None
    prev_time = None
    prev_velocities = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # current time in seconds

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            body_landmarks = {i: landmarks[i] for i in range(len(landmarks)) if i not in range(0, 11)}

            angles = {
                'RIGHT_SHOULDER': calculate_angle(landmarks[14], landmarks[12], landmarks[24]),
                'LEFT_SHOULDER': calculate_angle(landmarks[13], landmarks[11], landmarks[23]),
                'RIGHT_HIP': calculate_angle(landmarks[12], landmarks[24], landmarks[25]),
                'LEFT_HIP': calculate_angle(landmarks[11], landmarks[23], landmarks[26]),
                'RIGHT_ELBOW': calculate_angle(landmarks[14], landmarks[12], landmarks[16]),
                'LEFT_ELBOW': calculate_angle(landmarks[13], landmarks[11], landmarks[15]),
                'RIGHT_KNEE': calculate_angle(landmarks[24], landmarks[26], landmarks[28]),
                'LEFT_KNEE': calculate_angle(landmarks[23], landmarks[25], landmarks[27]),
            }

            phase = determine_phase(angles, landmarks)
            color = get_phase_color(phase)
            cv2.putText(frame, phase, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            draw_landmarks(frame, body_landmarks, mp_pose.POSE_CONNECTIONS, color)

            draw_angle(frame, landmarks[24], landmarks[12], landmarks[26], angles['RIGHT_SHOULDER'], color)
            draw_angle(frame, landmarks[23], landmarks[11], landmarks[25], angles['LEFT_SHOULDER'], color)
            draw_angle(frame, landmarks[12], landmarks[24], landmarks[26], angles['RIGHT_HIP'], color)
            draw_angle(frame, landmarks[11], landmarks[23], landmarks[25], angles['LEFT_HIP'], color)
            draw_angle(frame, landmarks[24], landmarks[26], landmarks[28], angles['RIGHT_KNEE'], color)
            draw_angle(frame, landmarks[23], landmarks[25], landmarks[27], angles['LEFT_KNEE'], color)
            draw_angle(frame, landmarks[13], landmarks[11], landmarks[15], angles['LEFT_ELBOW'], color)
            draw_angle(frame, landmarks[14], landmarks[12], landmarks[16], angles['LEFT_KNEE'], color)

            if prev_landmarks is not None and prev_time is not None:
                time_interval = current_time - prev_time
                velocities = {}
                accelerations = {}

                for point_name, landmark_index in [('RIGHT_ELBOW', 14), ('LEFT_ELBOW', 13),
                                                   ('RIGHT_KNEE', 26), ('LEFT_KNEE', 25),
                                                   ('RIGHT_HIP', 24), ('LEFT_HIP', 23),
                                                   ('RIGHT_SHOULDER', 12), ('LEFT_SHOULDER', 11)]:
                    
                    prev_pos = [prev_landmarks[landmark_index].x, prev_landmarks[landmark_index].y, prev_landmarks[landmark_index].z]
                    curr_pos = [landmarks[landmark_index].x, landmarks[landmark_index].y, landmarks[landmark_index].z]

                    velocity = calculate_velocity(prev_pos, curr_pos, time_interval)
                    velocities[point_name] = velocity

                    cv2.putText(frame, f'V_{point_name}: {np.linalg.norm(velocity):.2f}', (int(curr_pos[0] * frame.shape[1]), int(curr_pos[1] * frame.shape[0]) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

                    if prev_velocities is not None:
                        acceleration = calculate_acceleration(prev_velocities[point_name], velocity, time_interval)
                        accelerations[point_name] = acceleration

                        cv2.putText(frame, f'A_{point_name}: {np.linalg.norm(acceleration):.2f}', (int(curr_pos[0] * frame.shape[1]),
                                                                                                   int(curr_pos[1] * frame.shape[0]) - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
                                    cv2.LINE_AA)

                prev_velocities = velocities

            prev_landmarks = landmarks
            prev_time = current_time

            print("Angles:", angles)
            print("Phase:", phase)

        out.write(frame)

    cap.release()
    out.release()