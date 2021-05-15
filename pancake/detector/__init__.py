from .base_class import BaseModel
from .YOLOv5.yolov5_class import Yolov5Model 

MODEL_REGISTRY = BaseModel.get_subclasses()