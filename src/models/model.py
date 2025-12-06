import os
from pathlib import Path
from pydantic import BaseModel, Field
from keras.saving import load_model
from typing import Optional, Any, Union, List, Dict

# Get the directory where this file is located
MODELS_DIR = Path(__file__).parent

# Load Models
try:
    binary_model = load_model(str(MODELS_DIR / 'binaryfailure.keras'))
    failure_type_model = load_model(str(MODELS_DIR / 'multiclassfailtype.keras'))
    print("✓ Binary failure model loaded successfully")
    print("✓ Multiclass failure type model loaded successfully")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    binary_model = None
    failure_type_model = None

# Input Data Model
class MachineInput(BaseModel):
    product_id: str = Field(..., description="Product ID of the machine")
    type: str = Field(..., description="Machine type (e.g., L, M, H)")
    air_temperature: float = Field(..., description="Air temperature in Kelvin", ge=0)
    process_temperature: float = Field(..., description="Process temperature in Kelvin", ge=0)
    rotational_speed: float = Field(..., description="Rotational speed in RPM", ge=0)
    torque: float = Field(..., description="Torque in Nm", ge=0)
    tool_wear: float = Field(..., description="Tool wear in minutes", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "M14860",
                "type": "M",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 0.0
            }
        }

# Binary Prediction Response
class BinaryPredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction: 0 (not failed) or 1 (failed)")
    prediction_label: str = Field(..., description="Human readable label: 'not failed' or 'failed'")
    probability: float = Field(..., description="Probability of failure (0.0 to 1.0)")
    confidence: float = Field(..., description="Confidence score")
    input_data: dict = Field(..., description="Input data used for prediction")

# Multiclass Prediction Response
class MulticlassPredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted failure type as string")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each failure type")
    confidence: float = Field(..., description="Confidence score for the prediction")
    input_data: dict = Field(..., description="Input data used for prediction")
    ambiguous: bool = Field(False, description="Whether the model is uncertain about the prediction")
    top_k: Optional[List[Dict[str, Any]]] = Field(None, description="Top-K predicted classes with probabilities")
    suggested_override: Optional[Dict[str, Any]] = Field(None, description="Optional suggested override or rule-based hint")

# Generic Response Model
class GenericResponseModel(BaseModel):
    status_code: int
    message: str
    error: str = ""
    data: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True