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
def determine_phase(reference_angles, landmarks):

    # Catch angle limits
    CATCH_ELBOW = 10
    CATCH_KNEE = 45
    CATCH_HIP = 45

    # Drive angle limits
    DRIVE_ELBOW_MAX = 30
    DRIVE_KNEE_MIN = 150
    DRIVE_HIP_MAX = 90

    # Finish angle limits
    FINISH_ELBOW_MIN = 30
    FINISH_ELBOW_MAX = 50
    FINISH_KNEE = 120
    FINISH_HIP_MIN = 80
    FINISH_HIP_MAX = 110

    # Recovery angle limits
    RECOVERY_ELBOW_MAX = 20
    RECOVERY_KNEE_MIN = 30
    RECOVERY_KNEE_MAX = 140
    RECOVERY_HIP = 90

    # Function to get most visible angle from prediction
    def get_visible_angle(angle_right, angle_left, visibility_right, visibility_left):
        if visibility_right > visibility_left:
            return angle_right
        else:
            return angle_left
    
    # Getting most visible angle
    shoulder = get_visible_angle(reference_angles['shoulder']['right']['angle'], 
                                 reference_angles['shoulder']['left']['angle'], 
                                 reference_angles['shoulder']['right']['visibility'], 
                                 reference_angles['shoulder']['left']['visibility'])
    
    hip      = get_visible_angle(reference_angles['hip']['right']['angle'], 
                                 reference_angles['hip']['left']['angle'], 
                                 reference_angles['hip']['right']['visibility'], 
                                 reference_angles['hip']['left']['visibility'])
    
    elbow    = get_visible_angle(reference_angles['elbow']['right']['angle'], 
                                 reference_angles['elbow']['left']['angle'], 
                                 reference_angles['elbow']['right']['visibility'], 
                                 reference_angles['elbow']['left']['visibility'])
    
    knee     = get_visible_angle(reference_angles['knee']['right']['angle'], 
                                 reference_angles['knee']['left']['angle'], 
                                 reference_angles['knee']['right']['visibility'], 
                                 reference_angles['knee']['left']['visibility'])
    
    # Determine phase from angle comparison
    if elbow < CATCH_ELBOW and knee < CATCH_KNEE and hip < CATCH_HIP:
        return 'Catch'
    
    if elbow < DRIVE_ELBOW_MAX and knee < DRIVE_KNEE_MIN and hip < DRIVE_HIP_MAX:
        return 'Drive'
    
    if FINISH_ELBOW_MAX > elbow > FINISH_ELBOW_MIN and knee > FINISH_KNEE and FINISH_HIP_MIN < hip < FINISH_HIP_MAX:
        return 'Finish'
    
    if elbow < RECOVERY_ELBOW_MAX and RECOVERY_KNEE_MIN < knee < RECOVERY_KNEE_MAX and hip < RECOVERY_HIP:
        return 'Recovery'
    
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