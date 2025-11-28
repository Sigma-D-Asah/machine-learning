from http import HTTPStatus
from fastapi import APIRouter

from ..models import GenericResponseModel, MachineInput
from ..services.failure_prediction import FailurePredictionService

failure_prediction_router = APIRouter(
    prefix="/v1/failure",
    tags=["failure_prediction"],
)

# Definisikan route
@failure_prediction_router.post(
    "/predict",
    status_code=HTTPStatus.OK,
    response_model=GenericResponseModel,
)
async def predict_failure(request: MachineInput):
    try:
        response = await FailurePredictionService().predict_failure(request)
        return GenericResponseModel(
            status_code=HTTPStatus.OK,
            message="Success",
            error="",
            data=response,
        )
    except Exception as e:
        return GenericResponseModel(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            message="Internal Server Error",
            error=str(e),
            data=None,
        )