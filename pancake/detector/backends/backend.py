""" Pancake Backend Base Class """
from abc import ABC, abstractmethod

from typing import Tuple, Union, List

import numpy as np
import torch


class Backend(ABC):
    """ [_Abstract Class_] Base class of the Backends

    __Base Class__:
        All Backends to be used within this framework have to inherit from this class. \
        The inheritance will automatically register every subclass into the registry thus \
        allowing for modular access to the backend strategies.
    """

    _subclasses = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def get_subclasses(cls) -> dict:
        """Returns all subclasses of this base class.
        The dictionary poses as Backend registry.

        Returns:
            dict: Dictionary containing all child classes.
        """
        return dict(cls._subclasses)

    def __init_subclass__(cls):
        module_name = cls.__module__.split(".")[-1].replace("backend_", "")
        Backend._subclasses[module_name] = cls

    @abstractmethod
    def detect(
        self, img: Union[np.ndarray, List[np.ndarray]], *args, **kwargs
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Method to wrap the backend strategy.

        Args:
            img (Union[np.ndarray, List[np.ndarray]]): Image or list of Images in BGR [c, w, h]

        Raises:
            NotImplementedError: (this is an abstract class)

        Returns:
            Tuple[torch.Tensor, np.ndarray]:
                First member of the tuple is the detection matrix in [xyxy, conf, cls] and
                second, the resulting image from application of the strategy in BGR
                [c, w , h].
        """
        raise NotImplementedError
