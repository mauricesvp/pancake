"""Custom trained detector based on YOLOv5."""
import torch

import pancake.models as m
from pancake.utils.common import fix_path
from .detector import Detector


class YOLOCustomDetector(Detector):
    """Very simple detector using pretrained yolov5."""

    def __init__(self, config, *args, **kwargs) -> None:
        weights = config["weights"]
        weights_cfg = fix_path(weights)
        model = config["model"]
        conf_thres = float(config["conf_thres"])
        iou_thres = float(config["iou_thres"])
        classes = None if "None" == config["classes"] else config["classes"]
        agnostic_nms = True if "True" == config["agnostic_nms"] else False
        img_size = int(config["img_size"])
        device = kwargs.get("device", "CPU")
        if device.isdigit():
            device = int(device)

        self.model = m.MODEL_REGISTRY[model](
            device, weights_cfg, conf_thres, iou_thres, classes, agnostic_nms, img_size
        )

    def detect(self, imgs) -> list:
        if type(imgs) is not list:
            imgs = [imgs]
        res = self.model.infer(imgs)
        return res
