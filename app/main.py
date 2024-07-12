from services.video_services import process_video
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mostrar apenas erros

process_video('video/720p_2.mp4','outputs/output.mp4')
