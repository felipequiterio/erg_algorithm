from services.video_services import process_video
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mostrar apenas erros

# process_video('video/720p_2.mp4','outputs/output.mp4')

import uvicorn
import asyncio

# from utils.log import get_custom_logger

logger = get_custom_logger('SERVER')


if __name__ == '__main__':
    logger.info('Initializing server...')
    uvicorn.run("server:app", port=config.HTTP_PORT, host="0.0.0.0", reload=False)
