"""Tracker base class."""
from abc import ABC, abstractmethod

class BaseTracker(ABC):
    _subclasses = {}

    def __init__(self, *args, **kwargs) -> None:
        pass
    
    @classmethod
    def get_subclasses(cls):
        return dict(cls._subclasses)

    def __init_subclass__(cls):
        module_name = cls.__module__.split('_')[1]
        class_name = module_name if not module_name.endswith('class') else module_name[:6]
        BaseTracker._subclasses[class_name] = cls

    @abstractmethod
    def update(self, det):
        raise NotImplementedError

