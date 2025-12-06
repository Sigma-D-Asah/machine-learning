from http import HTTPStatus
from fastapi import APIRouter, HTTPException

from ..models.model import (
    GenericResponseModel, 
    MachineInput,
    BinaryPredictionResponse,
    MulticlassPredictionResponse
)
from ..services.failure_prediction import FailurePredictionService

# Initialize router
failure_prediction_router = APIRouter(
    prefix="/api/v1/failure",
    tags=["Failure Prediction"],
)

# Initialize service
prediction_service = FailurePredictionService()


@failure_prediction_router.get(
    "/health",
    status_code=HTTPStatus.OK,
    response_model=GenericResponseModel,
    summary="Health Check",
    description="Check if the failure prediction service is running and models are loaded"
)
async def health_check():
    """Health check endpoint to verify service status."""
    try:
        models_status = {
            "binary_model_loaded": prediction_service.binary_model is not None,
            "failure_type_model_loaded": prediction_service.failure_type_model is not None
        }
        
        return GenericResponseModel(
            status_code=HTTPStatus.OK,
            message="Service is healthy",
            error="",
            data=models_status
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@failure_prediction_router.post(
    "/predict/binary",
    status_code=HTTPStatus.OK,
    response_model=GenericResponseModel,
    summary="Binary Failure Prediction",
    description="Predict if machine will fail (0: not failed, 1: failed)"
)
async def predict_binary_failure(request: MachineInput):
    """
    Predict binary failure status.
    
    - **Returns**: 0 (not failed) or 1 (failed)
    - **Output**: Integer value with probability
    """
    try:
        result = prediction_service.predict_binary_failure(request)
        
        return GenericResponseModel(
            status_code=HTTPStatus.OK,
            message="Binary prediction successful",
            error="",
            data=result.dict()
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@failure_prediction_router.post(
    "/predict/type",
    status_code=HTTPStatus.OK,
    response_model=GenericResponseModel,
    summary="Failure Type Prediction",
    description="Predict the type of failure (multiclass classification)"
)
async def predict_failure_type(request: MachineInput):
    """
    Predict the type of failure.
    
    - **Returns**: String indicating the failure type
    - **Output**: Failure type as string with probabilities for all classes
    """
    try:
        # First check binary prediction — only run multiclass if binary predicts a failure
        binary_result = prediction_service.predict_binary_failure(request)
        if binary_result.prediction == 0:
            # If binary model predicts not failed, we won't run multiclass model
            # Return a clear response indicating multiclass was not performed
            # Build a probabilities dict with 'No Failure' probability = 1.0
            probs = {label: (1.0 if label == 'No Failure' else 0.0) for label in prediction_service.failure_type_labels}
            no_failure_response = {
                'prediction': 'No Failure',
                'probabilities': probs,
                'confidence': 1.0,
                'input_data': {
                    'product_id': request.product_id,
                    'type': request.type,
                    'air_temperature': request.air_temperature,
                    'process_temperature': request.process_temperature,
                    'rotational_speed': request.rotational_speed,
                    'torque': request.torque,
                    'tool_wear': request.tool_wear
                }
            }
            # Add fields to match MulticlassPredictionResponse
            no_failure_response['ambiguous'] = False
            no_failure_response['top_k'] = None
            no_failure_response['suggested_override'] = None
            return GenericResponseModel(
                status_code=HTTPStatus.OK,
                message="Binary predicted not failed; multiclass prediction not performed",
                error="",
                data=no_failure_response
            )

        result = prediction_service.predict_failure_type(request)
        
        return GenericResponseModel(
            status_code=HTTPStatus.OK,
            message="Failure type prediction successful",
            error="",
            data=result.dict()
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )