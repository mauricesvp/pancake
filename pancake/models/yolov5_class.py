""" Encapsulates YOLOv5 functionalities (loading, inference) """
from typing import List

import numpy as np
from typing import Type
import torch

from .base_class import BaseModel
from .experimental import attempt_load
from pancake.logger import setup_logger
from pancake.utils.general import check_img_size, non_max_suppression
from pancake.utils.function_profiler import profile

l = setup_logger(__name__)


class Yolov5Model(BaseModel):
    def __init__(
        self,
        device: str,
        weights: str,
        conf_thres: float,
        iou_thres: float,
        classes: List[int],
        agnostic_nms: bool,
        img_size: int,
        max_det: int,
        *args,
        **kwargs,
    ):
        """ Class facilitating all YOLOv5 functionalities.

        Args:
            device (str): Device to calculate on (CPU, GPU)
            weights (str): Path to custom trained weights or name of the official pretrained yolo
            conf_thres (float): Confidence threshold
            iou_thres (float): Intersection over union threshold
            classes (List[int]): Filter by class id
            agnostic_nms (bool): Enable class-agnostic NMS
            img_size (int): Specified input image size, will automatically resize and pad the image
            max_det (int): Max number of detections in an infered frame

        Note:
        - 'weights' parameter can contain either a Path or a name of an pretrained YOLOv5 architecture, \
            for a list of available models refer to: [here](https://github.com/ultralytics/yolov5/releases)
        """        
        super(Yolov5Model, self).__init__(device)
        # load model
        self.model = attempt_load(weights, map_location=self._device)
        self.model.eval()
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
        self._max_det = max_det

        self._required_img_size = check_img_size(img_size, self._stride)
        self._init_infer(self._required_img_size)

    def _init_infer(self, img_size: int):
        """ Does one forward pass on the NN for warmup of the GPU.

        Args:
            img_size (int): Padded and resized image size conforming with model stride
        """        
        super(Yolov5Model, self)._init_infer(img_size)

    def prep_image_infer(self, img: np.array) -> torch.Tensor:
        """ Preprocessing procedure for the images.

        Args:
            img (np.array): Padded and resized image (meeting stride-multiple constraints) \
                on [c, w, h] or [bs, c, w, h]

        Returns:
            torch.Tensor: Preprocessed image 4d tensor [bs, c, w, h] (on device, \
                expanded dim (,4), half precision (fp16))
        """        
        return super(Yolov5Model, self).prep_image_infer(img)

    def infer(self, img: np.array) -> List[torch.Tensor]:
        """ Inference method

        Args:
            img (np.array): Resized and padded image [c, w, h] or [bs, c, w, h]

        Returns:
            List[torch.Tensor]: List of detections, on (,6) tensor [xyxy, conf, cls]
        """        
        # Prepare img for inference
        img = self.prep_image_infer(img)

        # Inference
        pred = super(Yolov5Model, self).infer(img)

        # Apply NMS
        pred = non_max_suppression(
            pred,
            self._conf_thres,
            self._iou_thres,
            self._classes,
            self._agnostic_nms,
            False,
            self._max_det,
        )
        pred = [x.cpu() for x in pred]
        return pred, img
