"""Custom trained detector based on YOLOv5."""
import math

import cv2
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

    def round(self, val: int, base: int) -> int:
        return self.model._stride * math.floor(val / self.model._stride)

    def detect(self, imgs) -> list:
        # TODO: Make this better ...
        h, w, _ = imgs.shape
        self.model._stride = self.round(max(h, w) // 10, 64)
        self.model._stride = max(64, self.model._stride)
        hr = self.round(h, self.model._stride)
        wr = self.round(w, self.model._stride)
        if h != hr or w != wr:
            imgs = cv2.resize(imgs, (wr, hr))
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        res, _ = self.model.infer(imgs)
        return res
