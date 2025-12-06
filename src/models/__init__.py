# Define all variables
from .model import (
    MachineInput,
    BinaryPredictionResponse,
    MulticlassPredictionResponse,
    GenericResponseModel,
    binary_model,
    failure_type_model
)

__version__ = "0.1.0"

__all__ = [
    "MachineInput",
    "BinaryPredictionResponse",
    "MulticlassPredictionResponse",
    "GenericResponseModel",
    "binary_model",
    "failure_type_model"
]
