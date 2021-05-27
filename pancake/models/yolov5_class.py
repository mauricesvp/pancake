"""Encapsulates yolov5 functionalities (loading, inference)
"""
import numpy as np
from typing import Type
import torch

from .base_class import BaseModel
from .experimental import attempt_load
from pancake.logger import setup_logger
from pancake.utils.general import check_img_size, non_max_suppression

l = setup_logger(__name__)


class Yolov5Model(BaseModel):
    def __init__(
        self,
        device: str,
        weights: str,
        conf_thres: float,
        iou_thres: float,
        classes: int,
        agnostic_nms: bool,
        img_size: int,
        *args,
        **kwargs,
    ):
        """
        :param device (torch.device): device to calculate on (cpu, gpu)
        :param weights (str): path to custom trained weights or name of the official pretrained yolo
        :param conf_thres (float): confidence threshold
        :param iou_thres (float): intersection over union threshold
        :param classes (int): filter by class 0, or 0 2 3
        :param agnostic_nms(bool): class-agnostic NMS
        :param img_size (int): specified image size
        """
        super(Yolov5Model, self).__init__(device)
        # load model
        self.model = attempt_load(weights, map_location=self._device)
        if self._half:
            self.model.half()  # to FP16

        self._stride = int(self.model.stride.max())  # model stride
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )  # get class names

        l.debug(f"Class names: {self.names}")

        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._classes = classes
        self._agnostic_nms = agnostic_nms

        self._required_img_size = check_img_size(img_size, self._stride)
        self._init_infer(self._required_img_size)

    def _init_infer(self, img_size):
        """
        Does one forward pass on the network for initialization on gpu

        :param img_size: padded, resized image size
        """
        super(Yolov5Model, self)._init_infer(img_size)

    def prep_image_infer(self, img: Type[np.array]) -> Type[torch.Tensor]:
        """
        :param img: padded and resized image (meeting stride-multiple constraints)
        :return prep_img: preprocessed image 4d tensor [, R, G, B] (on device,
                          expanded dim (,4), half precision (fp16))
        """
        return super(Yolov5Model, self).prep_image_infer(img)

    def infer(self, img: Type[np.array]) -> Type[torch.Tensor]:
        """
        :param img (np.array): resized and padded image [R, G, B] or [, R, G, B]

        :return pred (tensor): list of detections, on (,6) tensor [xyxy, conf, cls]
                img (tensor): preprocessed image 4d tensor [, R, G, B] (on device,
                              expanded dim (,4), half precision (fp16))
        """
        # Prepare img for inference
        img = self.prep_image_infer(img)

        # Inference
        pred = super(Yolov5Model, self).infer(img)

        # Apply NMS
        pred = non_max_suppression(
            pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms
        )
        return pred, img
