"""Detector backend base class."""
from abc import ABC, abstractmethod


class Backend(ABC):
    _subclasses = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def get_subclasses(cls):
        return dict(cls._subclasses)

    def __init_subclass__(cls):
        module_name = cls.__module__.split(".")[-1].replace("backend_", "")
        Backend._subclasses[module_name] = cls

    @abstractmethod
    def detect(self, img, roi=None, *args, **kwargs) -> list:
        raise NotImplementedError
