"""Tracker based on Centroid tracking."""
from pancake.tracker.tracker import Tracker


class CentroidTracker(Tracker):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def track(self, rect) -> list:
        return []
