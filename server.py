from fastapi import FastAPI, Request

from routes import router
from utils.log import get_custom_logger

app = FastAPI(
    version="0.0.1",
    title="Erg Algorithm",
    description="Analysis algorithm",
)

# logger = get_custom_logger('MIDDLEWARE')

app.include_router(router)
