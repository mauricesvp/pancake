"""Tracker base class."""

class Tracker:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def track(self, rect) -> list:
        raise NotImplementedError
