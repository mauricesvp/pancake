"""Detector based on YOLOv5."""
import torch

from .detector import Detector


class YOLOSimpleDetector(Detector):
    """Very simple detector using pretrained yolov5."""

    def __init__(self, *args, **kwargs) -> None:
        self.model = torch.hub.load("ultralytics/yolov5", f"yolov5s", pretrained=True)

    def detect(self, imgs) -> list:
        if type(imgs) is not list:
            imgs = [imgs]
        res = self.model(imgs)
        return res.xyxy