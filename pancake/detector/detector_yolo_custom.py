""" Custom Detector Class based on YOLOv5 """
from typing import List

import math
import numpy as np

import torch

from .detector import Detector
import pancake.models as m

# from pancake.models.tensorrt.yolov5_tensorrt import Yolov5TRT
from pancake.logger import setup_logger
from pancake.utils.common import fix_path
from pancake.utils.datasets import letterbox
from pancake.utils.general import scale_coords
from pancake.utils.function_profiler import profile
from pancake.models.yolov5_class import Yolov5Model

l = setup_logger(__name__)


class YOLOCustomDetector(Detector):
    def __init__(self, config: dict, *args, **kwargs) -> None:
        """ This class encapsulates the YOLOv5 module.

        Description:
            During initialization the configurations are retrieved and parsed to the \
            initialization of the YOLOv5 class. \
            When the flag 'trt' is True, self.model is overwritten by the YOLOv5 TensorRT model.

        Args:
            config (dict): Configuration dictionary
        """        
        self.weights = config["weights"]
        weights_cfg = (
            fix_path(self.weights) if type(self.weights) is str else self.weights
        )
        model = config["model"]

        trt = config["trt"]
        trt_engine_path = config["trt_engine_path"]
        trt_plugin_library = config["trt_plugin_library"]

        conf_thres = float(config["conf_thres"])
        iou_thres = float(config["iou_thres"])
        classes = None if "None" == config["classes"] else config["classes"]
        agnostic_nms = True if "True" == config["agnostic_nms"] else False
        img_size = int(config["img_size"])
        device = kwargs.get("device", "CPU")
        max_det = int(config["max_det"])

        self.model: Yolov5Model = m.MODEL_REGISTRY[model](
            device,
            weights_cfg,
            conf_thres,
            iou_thres,
            classes,
            agnostic_nms,
            img_size,
            max_det,
        )

        try:
            if trt:
                from pancake.models.tensorrt.yolov5_trt_2 import Yolov5TRT

                self.model = (
                    Yolov5TRT(self.model, trt_engine_path, trt_plugin_library, device)
                    if trt
                    else self.model
                )
        except ModuleNotFoundError:
            l.info(f"Will fallback to weights file: {self.weights}")
            

    def detect(self, imgs: List[np.ndarray]) -> List[torch.Tensor]:
        """ Wrapper for detection calculation.

        Description:
            - Pads and resizes the images to conform with the model
            - Calls the infer method of underlying model in order to retrieve detections
            - Rescales the detections

        Args:
            imgs (List[np.ndarray]): List of ndarrays, images in BGR [bs, c, w, h]

        Returns:
            List[torch.Tensor]: List of tensors, detections on (,6) tensors [xyxy, conf, cls]
        """        
        pr_imgs = self._preprocess(imgs)
        img_sizes = [img.shape for img in imgs]

        # Inference
        # l.debug(f"Inference on: {pr_imgs.shape}")
        det, _ = self.model.infer(pr_imgs)

        res = self._postprocess(det, pr_imgs, img_sizes)
        return res

    def _preprocess(self, imgs: List[np.ndarray]) -> np.array:
        """ Pads and resizes the images, converts the images to RGB.

        Args:
            imgs (List[np.ndarray]): List of ndarrays, images in BGR [bs, c, w, h]

        Returns:
            np.array: Padded and resized images in RGB [bs, c, w, h]
        """        
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
        self, det: List[torch.Tensor], pr_imgs: np.array, img_sizes: list
    ) -> list:
        """ Rescales the detection matrix from padded and resized to
        the original size.

        Args:
            det (List[torch.Tensor]): Tensor list of detections, on (,6) tensor [bs, xyxy, conf, cls]
            pr_imgs (np.array): Preprocessed images [bs, c, w, h]
            img_sizes (list): List of original image sizes [bs, (w, h)]

        Returns:
            list: Tensor list of rescaled detections, on (,6) tensor [bs, xyxy, conf, cls]
        """ 
        # Rescale images from preprocessed to original
        res = [None] * len(det)
        for i, x in enumerate(det):
            x[:, :4] = scale_coords(pr_imgs.shape[2:], x[:, :4], img_sizes[i]).round()
            res[i] = x
        return res
