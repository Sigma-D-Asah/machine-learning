from fastapi import FastAPI
from ..controllers.failure_prediction.controller import failure_prediction_router

# Create app instance
app = FastAPI()

# Include router
app.include_router(failure_prediction_router)