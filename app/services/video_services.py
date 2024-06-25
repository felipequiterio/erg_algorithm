import cv2
import mediapipe as mp

from app.services.analysis_services import calculate_angle, determine_phase, get_phase_color, draw_landmarks


def process_video(video_path: str, output_path: str):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            body_landmarks = {i: landmarks[i] for i in range(len(landmarks)) if i not in range(0, 11)}

            angles = {
                'trunk_leg': calculate_angle(landmarks[11], landmarks[23], landmarks[25]),  # ombro, quadril, joelho
                'thigh_calf': calculate_angle(landmarks[23], landmarks[25], landmarks[27])  # quadril, joelho, tornozelo
            }

            phase = determine_phase(angles, landmarks)
            color = get_phase_color(phase)
            cv2.putText(frame, phase, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            draw_landmarks(frame, body_landmarks, mp_pose.POSE_CONNECTIONS, color)

            print("Angles:", angles)
            print("Phase:", phase)

        out.write(frame)

    cap.release()
    out.release()