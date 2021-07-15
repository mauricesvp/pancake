""" Detector based on YOLOv5 """
from typing import List

import torch
import numpy as np

from .detector import Detector


class YOLOSimpleDetector(Detector):
    """ Very simple detector using pretrained YOLOv5 from torch.hub """

    def __init__(self, *args, **kwargs) -> None:
        self.model = torch.hub.load("ultralytics/yolov5", f"yolov5s", pretrained=True)

    def detect(self, imgs: List[np.ndarray]) -> list:
        """ Encapsulates the detection procedure.

        Args:
            imgs (List[np.ndarray]): List of images of shape [bs, channels, width, height]

        Returns:
            list: [description]
        """        
        if type(imgs) is not list:
            imgs = [imgs]
        res = self.model(imgs)
        return res.xyxy
