import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
from ..models.model import (
    MachineInput, 
    binary_model, 
    failure_type_model,
    BinaryPredictionResponse,
    MulticlassPredictionResponse
)


class FailurePredictionService:
    def __init__(self):
        self.binary_model = binary_model
        self.failure_type_model = failure_type_model
        
        # Load scaler for normalization (same as training)
        scaler_path = Path(__file__).parent.parent / 'models' / 'scaler.pkl'
        try:
            self.scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load scaler: {e}")
            self.scaler = None
        
        # Mapping untuk failure types (sesuai dengan LabelEncoder dari training)
        # Order: alphabetical dari LabelEncoder
        self.failure_type_labels = [
            "Heat Dissipation Failure",     # 0
            "No Failure",                    # 1
            "Overstrain Failure",            # 2
            "Power Failure",                 # 3
            "Random Failures",               # 4
            "Tool Wear Failure"              # 5
        ]
        # Multiclass behavior config
        # Confidence threshold for accepting multiclass prediction (0.0 - 1.0)
        self.multiclass_confidence_threshold = 0.3
        # Rule-based suggestions (optional)
        self.rule_override_thresholds = {
            'tool_wear': 200  # minutes: if tool_wear >= this value, suggest Tool Wear Failure
        }
        
    def _prepare_input_features(self, machine_input: MachineInput) -> np.ndarray:
        """
        Convert MachineInput to numpy array for prediction.
        Uses label encoding for machine type (H=0, L=1, M=2) as per training.
        Applies MinMaxScaler normalization (same as training).
        
        IMPORTANT: Column order must match training data:
        ['type', 'air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
        """
        # Label encoding for type (from LabelEncoder on alphabetical order)
        # H=0, L=1, M=2 (alphabetical: H, L, M)
        type_mapping = {'H': 0, 'L': 1, 'M': 2}
        type_encoded = type_mapping.get(machine_input.type.upper(), 2)  # Default to M if unknown
        
        # Create feature array with 6 features (BEFORE normalization)
        # Order MUST match training: type, air_temperature, process_temperature, rotational_speed, torque, tool_wear
        features = np.array([[
            type_encoded,
            machine_input.air_temperature,
            machine_input.process_temperature,
            machine_input.rotational_speed,
            machine_input.torque,
            machine_input.tool_wear
        ]], dtype=np.float32)
        
        # Apply MinMaxScaler normalization (critical for correct predictions)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        else:
            raise Exception("Scaler not loaded. Cannot normalize input features.")
        
        return features
    
    def predict_binary_failure(self, machine_input: MachineInput) -> BinaryPredictionResponse:
        """
        Predict binary failure (0: not failed, 1: failed).
        
        Args:
            machine_input: MachineInput object containing machine parameters
            
        Returns:
            BinaryPredictionResponse with prediction and probability
        """
        if self.binary_model is None:
            raise Exception("Binary model is not loaded")
        
        try:
            # Prepare input features
            features = self._prepare_input_features(machine_input)
            
            # Make prediction
            prediction_prob = self.binary_model.predict(features, verbose=0)
            failure_probability = float(prediction_prob[0][0])
            
            # Binary prediction (0 or 1) with lower threshold for better sensitivity
            # Threshold 0.1 instead of 0.5 to catch more potential failures (especially for imbalanced data)
            prediction_binary = 1 if failure_probability >= 0.05 else 0
            
            # Translate to human-readable label
            prediction_label = "failed" if prediction_binary == 1 else "not failed"
            
            # Calculate confidence
            confidence = failure_probability if prediction_binary == 1 else (1 - failure_probability)
            
            return BinaryPredictionResponse(
                prediction=prediction_binary,
                prediction_label=prediction_label,
                probability=round(failure_probability, 4),
                confidence=round(confidence, 4),
                input_data={
                    'product_id': machine_input.product_id,
                    'type': machine_input.type,
                    'air_temperature': machine_input.air_temperature,
                    'process_temperature': machine_input.process_temperature,
                    'rotational_speed': machine_input.rotational_speed,
                    'torque': machine_input.torque,
                    'tool_wear': machine_input.tool_wear
                }
            )
            
        except Exception as e:
            raise Exception(f"Binary prediction failed: {str(e)}")
    
    def predict_failure_type(self, machine_input: MachineInput) -> MulticlassPredictionResponse:
        """
        Predict failure type (multiclass classification).
        Returns string output for the failure type.
        
        Args:
            machine_input: MachineInput object containing machine parameters
            
        Returns:
            MulticlassPredictionResponse with predicted failure type as string
        """
        if self.failure_type_model is None:
            raise Exception("Failure type model is not loaded")
        
        try:
            # Prepare input features
            features = self._prepare_input_features(machine_input)
            
            # Make prediction
            prediction_probs = self.failure_type_model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction_probs[0])
            
            # Get predicted failure type as string
            predicted_failure_type = self.failure_type_labels[predicted_class]
            
            # Create probabilities dictionary
            probabilities = {
                label: round(float(prob), 4) 
                for label, prob in zip(self.failure_type_labels, prediction_probs[0])
            }
            
            # Calculate confidence (max probability)
            confidence = float(np.max(prediction_probs[0]))

            # Create sorted top-k predictions (top 3)
            top_k = sorted(
                [
                    {"label": label, "prob": float(prob)}
                    for label, prob in zip(self.failure_type_labels, prediction_probs[0])
                ],
                key=lambda x: x['prob'],
                reverse=True
            )[:3]

            ambiguous = confidence < self.multiclass_confidence_threshold

            # Suggested override rule-based: if tool_wear is very high, suggest Tool Wear Failure
            suggested_override = None
            tw_threshold = self.rule_override_thresholds.get('tool_wear')
            if tw_threshold is not None and machine_input.tool_wear >= tw_threshold and predicted_failure_type != 'Tool Wear Failure':
                suggested_override = {
                    'label': 'Tool Wear Failure',
                    'reason': f'tool_wear >= {tw_threshold} (raw value: {machine_input.tool_wear})'
                }
            
            return MulticlassPredictionResponse(
                prediction=predicted_failure_type,
                probabilities=probabilities,
                confidence=round(confidence, 4),
                input_data={
                    'product_id': machine_input.product_id,
                    'type': machine_input.type,
                    'air_temperature': machine_input.air_temperature,
                    'process_temperature': machine_input.process_temperature,
                    'rotational_speed': machine_input.rotational_speed,
                    'torque': machine_input.torque,
                    'tool_wear': machine_input.tool_wear
                }
                ,
                ambiguous=ambiguous,
                top_k=top_k,
                suggested_override=suggested_override
            )
            
        except Exception as e:
            raise Exception(f"Multiclass prediction failed: {str(e)}")
    
