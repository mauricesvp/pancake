from .backend import Backend
from .backend_dei import DEI
from .backend_simple import SIMPLE

__all__ = ["BACKEND_REGISTRY", "setup_backend"]

BACKEND_REGISTRY = Backend.get_subclasses()


def setup_backend(config, detector):
    name = config.DETECTOR.BACKEND

    ROI = config.DATA.ROI
    return BACKEND_REGISTRY[name](detector, roi=ROI)
