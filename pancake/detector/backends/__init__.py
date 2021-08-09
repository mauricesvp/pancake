""" Package containing backend related modules. """
from typing import Type

from .backend import Backend
from .backend_dei import DEI
from .backend_simple import SIMPLE
from ..detector import Detector

__all__ = ["BACKEND_REGISTRY", "setup_backend"]

BACKEND_REGISTRY = Backend.get_subclasses()


def setup_backend(config: dict, detector: Type[Detector]) -> Type[Backend]:
    """Helper function to set up a backend strategy specified in the configurations.

    Args:
        config (dict): Dictionary containing configurations.
        detector (Type[Detector]): A detector subclass instance to infer the frames with.

    Returns:
        Type[Backend]: A Backend subclass instance.
    """
    assert detector

    name = config.BACKEND.NAME
    try:
        params = getattr(config.BACKEND, name.upper())
    except:
        params = None

    ROI = config.DATA.ROI
    return BACKEND_REGISTRY[name](detector, roi=ROI, config=params)
