"""Detector base class."""

class Detector:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def detect(self, img) -> list:
        raise NotImplementedError
