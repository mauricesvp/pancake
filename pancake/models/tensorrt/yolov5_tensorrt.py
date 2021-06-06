import numpy as np
import os
import pkg_resources
import torch
import torch.nn as nn
from typing import Type

from pancake.logger import setup_logger
from ..base_class import BaseModel
from .trt_utils import export_onnx

l = setup_logger(__name__)

for package in ['tensorrt']:
    try:
        dist = pkg_resources.get_distribution(package)
        l.info(u'\u2713 ' + '{} ({}) is installed'.format(dist.key, dist.version)) 
        import tensorrt
        trt_installed = True
    except pkg_resources.DistributionNotFound:
        l.info(u'\u2620 ' + '{} is NOT installed'.format(package))
        trt_installed = False


class Yolov5TRT(BaseModel):

    def __init__(self, yolov5, weights_path):
        # if trt not available return standard yolov5 model
        if not trt_installed:
            return yolov5

        self.yolov5 = yolov5

        # initialize model for export
        tmp_model = Yolov5TRT._init_export(
            self.yolov5.model
        )

        # TRT currently only supports non-batch inference
        batch_size = 1
        # input size (x, x)
        x = self.yolov5._required_img_size
        # whether half precision is supported
        input_tensor = (
            torch.zeros(batch_size, 3, x, x).float() 
            if not self.yolov5._half
            else torch.zeros(batch_size, 3, x, x).half()
        )
        # on cpu/gpu
        input_tensor = input_tensor.to(self.yolov5._device)

        onnx_path = (
            weights_path.replace(".pt", ".onnx")
            if type(weights_path) is str
            else weights_path[0].replace(".pt", ".onnx")
        )

        weights_name = onnx_path.split("/")[-1].split(".")[0]
        
        l.info(f"Converting PyTorch model from weights {weights_name} to ONNX")
        export_onnx(tmp_model, onnx_path, input_tensor)
        
        if not os.path.isfile(onnx_path):
            l.info("Couldn't convert to ONNX, returning standard Yolov5")
            return self.yolov5
        
    
    @staticmethod
    def _init_export(model):
        from ...utils.activations import Hardswish, SiLU
        from ..yolo import Detect
        from ..common import Conv

        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif isinstance(m, Detect):
                m.inplace = opt.inplace
                m.onnx_dynamic = opt.dynamic
            # m.forward = m.forward_export  # assign forward (optional)
        return model

    def _init_infer(self, img_size):
        """
        Does one forward pass on the network for initialization on gpu

        :param img_size: padded, resized image size
        """
        pass


    def prep_image_infer(self, img: Type[np.array]) -> Type[torch.Tensor]:
        """
        Preprocesses images for inference (on device, expanded dim (,4), half precision (fp16), normalized)

        :param img: padded and resized image
        :return prep_img: preprocessed image
        """
        pass


    def infer(self, img: Type[np.array]) -> Type[torch.Tensor]:
        """
        :param img (np.array): resized and padded image [R, G, B] or [, R, G, B]

        :return pred (tensor): list of detections, on (,6) tensor [xyxy, conf, cls]
                img (tensor): preprocessed image 4d tensor [, R, G, B] (on device,
                              expanded dim (,4), half precision (fp16))
        """
        pass