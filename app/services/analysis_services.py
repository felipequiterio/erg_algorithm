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

# -------------------------- REFACTOR --------------------------------
def determine_phase(angles, landmarks):
    # Definindo os ângulos para as fases
    CATCH_ELBOW = 90
    CATCH_KNEE = 90
    CATCH_HIP = 90

    DRIVE_ELBOW_MIN = 90
    DRIVE_ELBOW_MAX = 180
    DRIVE_KNEE = 90
    DRIVE_HIP = 90

    FINISH_ELBOW = 90
    FINISH_KNEE = 90
    FINISH_HIP_MIN = 90
    FINISH_HIP_MAX = 180

    RECOVERY_ELBOW_MIN = 90
    RECOVERY_ELBOW_MAX = 180
    RECOVERY_KNEE = 90
    RECOVERY_HIP = 90

    def get_visible_angle(angle_right, angle_left, visibility_right, visibility_left):
        if visibility_right > visibility_left:
            return angle_right
        else:
            return angle_left
    
    right_shoulder = angles['RIGHT_SHOULDER']
    left_shoulder = angles['LEFT_SHOULDER']
    
    right_hip = angles['RIGHT_HIP']
    left_hip = angles['LEFT_HIP']
    
    right_elbow = angles['RIGHT_ELBOW']
    left_elbow = angles['LEFT_ELBOW']
    
    right_knee = angles['RIGHT_KNEE']
    left_knee = angles['LEFT_KNEE']
    
    visibility_right_shoulder = landmarks[12].visibility
    visibility_left_shoulder = landmarks[11].visibility
    
    visibility_right_hip = landmarks[24].visibility
    visibility_left_hip = landmarks[23].visibility
    
    visibility_right_elbow = landmarks[14].visibility
    visibility_left_elbow = landmarks[13].visibility
    
    visibility_right_knee = landmarks[26].visibility
    visibility_left_knee = landmarks[25].visibility
    
    shoulder = get_visible_angle(right_shoulder, left_shoulder, visibility_right_shoulder, visibility_left_shoulder)
    hip = get_visible_angle(right_hip, left_hip, visibility_right_hip, visibility_left_hip)
    elbow = get_visible_angle(right_elbow, left_elbow, visibility_right_elbow, visibility_left_elbow)
    knee = get_visible_angle(right_knee, left_knee, visibility_right_knee, visibility_left_knee)
    
    # Determinando a fase
    if elbow < CATCH_ELBOW and knee < CATCH_KNEE and hip < CATCH_HIP:
        return 'Catch'
    elif DRIVE_ELBOW_MIN < elbow < DRIVE_ELBOW_MAX and knee < DRIVE_KNEE and hip < DRIVE_HIP:
        return 'Drive'
    elif elbow > FINISH_ELBOW and knee > FINISH_KNEE and FINISH_HIP_MIN < hip < FINISH_HIP_MAX:
        return 'Finish'
    elif RECOVERY_ELBOW_MIN < elbow < RECOVERY_ELBOW_MAX and knee < RECOVERY_KNEE and hip > RECOVERY_HIP:
        return 'Recovery'
    else:
        return 'Unlabeled'

# -------------------------- REFACTOR --------------------------------

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


def draw_angle(image, point_a, point_b, point_c, angle, color=(255, 255, 255)):
    x1, y1 = int(point_a.x * image.shape[1]), int(point_a.y * image.shape[0])
    x2, y2 = int(point_b.x * image.shape[1]), int(point_b.y * image.shape[0])
    x3, y3 = int(point_c.x * image.shape[1]), int(point_c.y * image.shape[0])

    cv2.putText(image, str(int(angle)), (x2, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    

def calculate_velocity(pos1, pos2, time_interval):
    return (np.array(pos2) - np.array(pos1)) / time_interval

def calculate_acceleration(vel1, vel2, time_interval):
    return (np.array(vel2) - np.array(vel1)) / time_interval