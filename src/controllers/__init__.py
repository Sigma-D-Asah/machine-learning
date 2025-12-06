"""Controller package initializer

Exports the router defined in `failure_prediction_controller.py`.
"""
from .failure_prediction_controller import failure_prediction_router

__all__ = ["failure_prediction_router"]
__version__ = "0.1.0"