from .yolov5_class import Yolov5Model
from .base_class import BaseModel

MODEL_REGISTRY = BaseModel.get_subclasses()