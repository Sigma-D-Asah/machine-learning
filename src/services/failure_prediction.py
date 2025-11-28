import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..models import MachineInput, Model

class FailurePredictionService:
    def predict_failure(self, input: MachineInput) -> int:
        model = Model
        try:
            prediction = model.predict([input])
            return prediction
        except Exception as e:
            raise e
        