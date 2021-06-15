"""Custom trained detector based on YOLOv5."""
import math
import numpy as np
import time

import cv2
import torch

from .detector import Detector
import pancake.models as m

# from pancake.models.tensorrt.yolov5_tensorrt import Yolov5TRT
from pancake.models.tensorrt.yolov5_trt_2 import Yolov5TRT
from pancake.logger import setup_logger
from pancake.utils.common import fix_path
from pancake.utils.datasets import letterbox
from pancake.utils.general import scale_coords

l = setup_logger(__name__)


class YOLOCustomDetector(Detector):
    """Very simple detector using pretrained yolov5."""

    def __init__(self, config, *args, **kwargs) -> None:
        weights = config["weights"]
        weights_cfg = fix_path(weights) if type(weights) is str else weights
        model = config["model"]

        trt = config["trt"]
        trt_engine_path = config["trt_engine_path"]
        trt_plugin_library = config["trt_plugin_library"]
        nms_gpu = config["nms_gpu"]

        conf_thres = float(config["conf_thres"])
        iou_thres = float(config["iou_thres"])
        classes = None if "None" == config["classes"] else config["classes"]
        agnostic_nms = True if "True" == config["agnostic_nms"] else False
        img_size = int(config["img_size"])
        device = kwargs.get("device", "CPU")

        self.model = m.MODEL_REGISTRY[model](
            device, weights_cfg, conf_thres, iou_thres, classes, agnostic_nms, img_size
        )

        self.model = (
            Yolov5TRT(self.model, trt_engine_path, trt_plugin_library, nms_gpu)
            if trt
            else self.model
        )

    def round(self, val: int, base: int) -> int:
        return self.model._stride * math.floor(val / self.model._stride)

    def detect(self, imgs: list) -> list:
        """
        :param imgs (list): list of images, images as np.array in BGR
        :return res (list): tensor list of detections, on (,6) tensor [xyxy, conf, cls]
        """
        pr_imgs = self._preprocess(imgs)
        img_sizes = [img.shape for img in imgs]

        # Inference
        l.info(f"Inference on: {pr_imgs.shape}")
        det, _ = self.model.infer(pr_imgs)

        res = self._postprocess(det, pr_imgs, img_sizes)
        return res

    def _preprocess(self, imgs: list) -> np.array:
        if type(imgs) is not list:
            imgs = [imgs]

        # Padded resize
        pr_imgs = [
            letterbox(x, self.model._required_img_size, stride=self.model._stride)[0]
            for x in imgs
        ]

        # Stack
        pr_imgs = np.stack(pr_imgs, 0)

        # Convert
        pr_imgs = pr_imgs[:, :, :, ::-1].transpose(
            0, 3, 1, 2
        )  # BGR to RGB, to bsx3x416x416

        pr_imgs = np.ascontiguousarray(pr_imgs)
        return pr_imgs

    def _postprocess(
        self, det: torch.Tensor, pr_imgs: np.array, img_sizes: list
    ) -> list:
        # Rescale images from preprocessed to original
        res = [None] * len(det)
        for i, x in enumerate(det):
            x[:, :4] = scale_coords(pr_imgs.shape[2:], x[:, :4], img_sizes[i]).round()
            res[i] = x
        return res
