from .detector import Detector
from .detector_yolo_custom import YOLOCustomDetector
from .detector_yolo_simple import YOLOSimpleDetector

DETECTOR_REGISTRY = Detector.get_subclasses()
