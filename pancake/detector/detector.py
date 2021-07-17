""" Pancake Detector Base Class """
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch


class Detector(ABC):
    """ [_Abstract Class_] Base class of the Detectors
    
    __Base Class__:
        All Detectors to be used within this framework have to inherit from this class. \
        The inheritance will automatically register every subclass into the registry thus \
        allowing for modular access to the detectors.
    """    
    _subclasses = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def get_subclasses(cls) -> dict:
        """ Returns all subclasses of this base class. 
        The dictionary poses as Detector registry.

        Returns:
            dict: Dictionary containing all child classes.
        """        
        return dict(cls._subclasses)

    def __init_subclass__(cls):
        module_name = cls.__module__.split(".")[-1].replace("detector_", "")
        Detector._subclasses[module_name] = cls

    @abstractmethod
    def detect(self, img: np.ndarray, *args, **kwargs) -> List[torch.Tensor]:
        """ Method to encapsulate the detection procedure.

        Args:
            img (np.ndarray): List of ndarrays, images in BGR [batch size, channels, width, height]

        Raises:
            NotImplementedError: (this is an abstract class)

        Returns:
            List[torch.Tensor]: Tensor list of detections, on (,6) tensor [xyxy, conf, cls]
        """        
        raise NotImplementedError
