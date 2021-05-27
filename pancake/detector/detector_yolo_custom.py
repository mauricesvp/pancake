"""Custom trained detector based on YOLOv5."""
import math
import numpy as np

import cv2
import torch

import pancake.models as m
from pancake.utils.common import fix_path
from .detector import Detector
from pancake.utils.datasets import letterbox
from pancake.utils.general import scale_coords


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

        self.model = m.MODEL_REGISTRY[model](
            device, weights_cfg, conf_thres, iou_thres, classes, agnostic_nms, img_size
        )

    def round(self, val: int, base: int) -> int:
        return self.model._stride * math.floor(val / self.model._stride)

    def detect(self, imgs) -> list:
        # TODO: Make this better ...
         # Padded resize
        pr_imgs = letterbox(
            imgs, self.model._required_img_size, stride=self.model._stride)[0]

        # Convert
        imgs = imgs[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        pr_imgs = pr_imgs[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        pr_imgs = np.ascontiguousarray(pr_imgs)

        res = self.model.infer(pr_imgs)[0][0]
        res[:, :4] = scale_coords(
                    pr_imgs.shape[1:], res[:, :4], imgs.shape
                ).round()
        return [res]
