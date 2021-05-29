from .tracker import BaseTracker
from .tracker_deepsort import DEEPSORT
from .tracker_centroid import CentroidTracker

TRACKER_REGISTRY = BaseTracker.get_subclasses()
