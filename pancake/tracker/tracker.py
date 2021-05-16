"""Tracker base class."""
from abc import ABC, abstractmethod

class Tracker(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def update(self, det):
        raise NotImplementedError

