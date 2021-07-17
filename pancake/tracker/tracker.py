""" Pancake Tracker Base Class """
from abc import ABC, abstractmethod
import torch


class BaseTracker(ABC):
    """ [_Abstract Class_] Base class of the Trackers
    
    __Base Class__:
        All tracking algorithms to be used within this framework have to inherit from this class. \
        The inheritance will automatically register every subclass into the registry thus \
        allowing for modular access to the detectors.
    """    
    _subclasses = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def get_subclasses(cls):
        """ Returns all subclasses of this base class. 
        The dictionary poses as Tracker registry.

        Returns:
            dict: Dictionary containing all child classes.
        """      
        return dict(cls._subclasses)

    def __init_subclass__(cls):
        module_name = cls.__module__.split("_")[1]
        class_name = (
            module_name if not module_name.endswith("class") else module_name[:6]
        )
        BaseTracker._subclasses[class_name] = cls

    @abstractmethod
    def update(self, det: torch.Tensor, *args, **kwargs):
        """ Updates the internal tracker state.

        Args:
            det (torch.Tensor): Detections on (,6) tensor [xyxy, conf, cls]

        Raises:
            NotImplementedError: (this is an abtract method)
        """        
        raise NotImplementedError
