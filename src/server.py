from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from pathlib import Path

# Allow running server.py as a script (python src/server.py) or as a module (python -m src.server)
# Ensure project root is in sys.path so imports like `from src.controllers...` succeed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.failure_prediction_controller import failure_prediction_router

# Create FastAPI app instance
app = FastAPI(
    title="Machine Failure Prediction API",
    description="API for predicting machine failures using binary and multiclass models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(failure_prediction_router)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Machine Failure Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/failure/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )