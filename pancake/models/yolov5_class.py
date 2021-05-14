""" 
    encapsulates yolov5 functionalities (loading, inference)
"""
import torch
import sys
sys.path.append('src/models')

from .base_class import BaseModel
from utils.general import non_max_suppression

class Yolov5Model(BaseModel):
    def __init__(self, 
                 device: str, 
                 weights: str, 
                 conf_thres: float,
                 iou_thres: float,
                 classes: int,
                 agnostic_nms: bool,
                *args, **kwargs):
        """
        :param device (torch.device): device to calculate on (cpu, gpu)
        :param weights (str): path to custom trained weights or name of the official pretrained yolo
        :param conf_thres (float): confidence threshold
        :param iou_thres (float): intersection over union threshold
        :param classes (int): filter by class 0, or 0 2 3
        :param agnostic_nms(bool): class-agnostic NMS
        """
        super(Yolov5Model, self).__init__(device, weights, conf_thres, iou_thres, classes)
        self._agnostic_nms = agnostic_nms

    def _init_infer(self, img_size):
        """
        Does one forward pass on the network for initialization on gpu

        :param img_size: padded, resized image size
        """
        super(Yolov5Model, self)._init_infer(img_size)

    def prep_image_infer(self, img):
        """
        :param img: padded and resized image
        :return prep_img: preprocessed image (on device, expanded dim (,4), half precision (fp16))
        """
        return super(Yolov5Model, self).prep_image_infer(img)

    def infer(self, img):
        """
        :param img (tensor): resized and padded image preprocessed for inference (meeting stride-multiple constraints), 
                            4d tensor [x, R, G, B]
        :return pred (tensor): list of detections, on (,6) tensor [xyxy, conf, cls] 
        """
        # Inference
        pred = super(Yolov5Model, self).infer(img)

        # Apply NMS
        pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms)
        return pred
