from typing import Type

import torch
from .detector import Detector
from .detector_yolo_custom import YOLOCustomDetector
from .detector_yolo_simple import YOLOSimpleDetector

__all__ = ["DETECTOR_REGISTRY", "setup_detector"]


DETECTOR_REGISTRY = Detector.get_subclasses()


def setup_detector(config: dict) -> Type[Detector]:
    """ Helper function to set up a detector specified in the configurations.

    Description:
        Retrieves the configs, sets the device if possible and 
        initializes a detector from the detector registry.

    Args:
        config (dict): Dictionary containing configurations.

    Returns:
        Type[Detector]: A Detector subclass instance.
    """    
    name = config.DETECTOR.NAME
    params = getattr(config.DETECTOR, name.upper())

    device_cfg = config.DEVICE
    device = "CPU"

    if type(device_cfg) is str and torch.cuda.is_available():
        if device_cfg.upper() == "GPU":
            device = "0"
            import cv2

            cv2.cuda.setDevice(int(device))
        elif device_cfg.isdigit():
            device = device_cfg

    DETECTOR = DETECTOR_REGISTRY[name](params, device=device)
    return DETECTOR
