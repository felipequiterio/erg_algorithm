import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ab = b - a
    bc = b - c
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def determine_phase(angles, landmarks):
    trunk_leg = angles['trunk_leg']
    thigh_calf = angles['thigh_calf']
    hand_y = landmarks[15].y
    foot_y = landmarks[29].y

    if trunk_leg < 45 and thigh_calf < 45 and hand_y > foot_y:
        return 'Catch'
    elif 45 <= trunk_leg < 80 and 45 <= thigh_calf < 80:
        return 'Drive'
    elif trunk_leg >= 80 and thigh_calf >= 80 and hand_y < foot_y:
        return 'Finish'
    else:
        return 'Recovery'


def get_phase_color(phase):

    colors = {
        'Catch': (255, 0, 0),  # Blue
        'Drive': (0, 255, 0),  # Green
        'Finish': (0, 0, 255),  # Red
        'Recovery': (255, 255, 0)  # Yellow
    }
    return colors.get(phase, (255, 255, 255))  # White as default


def draw_landmarks(image, landmarks, connections, color):
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in landmarks and end_idx in landmarks:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(image, (int(start.x * image.shape[1]), int(start.y * image.shape[0])),
                     (int(end.x * image.shape[1]), int(end.y * image.shape[0])), color, 2)
            cv2.circle(image, (int(start.x * image.shape[1]), int(start.y * image.shape[0])), 5, color, -1)
            cv2.circle(image, (int(end.x * image.shape[1]), int(end.y * image.shape[0])), 5, color, -1)
