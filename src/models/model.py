from dataclasses import dataclass
import tensorflow as tf
from keras.saving import load_model

# Load Model
try:
    Model = load_model('predictive_maintenance.h5')
except Exception as e:
    print(f"Error saat memuat model: {e}")

# Define Input Data
@dataclass
class MachineInput:
    product_id: str
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float

# Generic Response Model
@dataclass
class GenericResponseModel:
    status_code: int
    message: str
    error: str
    data: Any