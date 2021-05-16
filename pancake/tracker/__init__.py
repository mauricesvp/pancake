from .tracker import BaseTracker
from .tracker_deepsort import DEEPSORT

TRACKER_REGISTRY = BaseTracker.get_subclasses()