angles = {
                'trunk_leg':    calculate_angle(landmarks[11], landmarks[23], landmarks[25]),  # ombro, quadril, joelho
                'thigh_calf':   calculate_angle(landmarks[23], landmarks[25], landmarks[27]),  # quadril, joelho, tornozelo
                'upper_arm':    calculate_angle(landmarks[11], landmarks[13], landmarks[15]),  # ombro, cotovelo, pulso
                'lower_arm':    calculate_angle(landmarks[13], landmarks[15], landmarks[17]),  # cotovelo, pulso, dedo
                'upper_leg':    calculate_angle(landmarks[23], landmarks[25], landmarks[27]),  # quadril, joelho, tornozelo
                'lower_leg':    calculate_angle(landmarks[25], landmarks[27], landmarks[31]),  # joelho, tornozelo, pé
                'upper_trunk':  calculate_angle(landmarks[11], landmarks[0] , landmarks[23]),  # ombro, pescoço, quadril
                'lower_trunk':  calculate_angle(landmarks[0] , landmarks[11], landmarks[13]),  # pescoço, ombro, cotovelo
                'hip_angle':    calculate_angle(landmarks[11], landmarks[23], landmarks[27]),  # ombro, quadril, tornozelo
                'hip_knee':     calculate_angle(landmarks[0] , landmarks[23], landmarks[25])  # pescoço, quadril, joelho
            }

angles = {

    'RIGHT_SHOULDER': calculate_angle(landmarks[14], landmarks[12], landmarks[24]),
    'LEFT_SHOULDER': calculate_angle(landmarks[13], landmarks[11], landmarks[23]),

    'RIGHT_HIP': calculate_angle(landmarks[12], landmarks[24], landmarks[25]),
    'LEFT_HIP': calculate_angle(landmarks[11], landmarks[23], landmarks[26]),

    'RIGHT_ELBOW': calculate_angle(landmarks[14], landmarks[12], landmarks[16]),
    'LEFT_ELBOW': calculate_angle(landmarks[13], landmarks[11], landmarks[15]),

    'RIGHT_KNEE': calculate_angle(landmarks[13], landmarks[11], landmarks[23]),
    'LEFT_KNEE': calculate_angle(landmarks[13], landmarks[11], landmarks[23]),

}