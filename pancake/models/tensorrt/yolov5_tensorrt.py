import numpy as np
import torch
from typing import Type
import pkg_resources

for package in ['tensorrt']:
    try:
        dist = pkg_resources.get_distribution(package)
        print(u'\u2713 ' + '{} ({}) is installed'.format(dist.key, dist.version)) 
        import tensorrt
    except pkg_resources.DistributionNotFound:
        print(u'\u2620 ' + '{} is NOT installed'.format(package))
        raise ModuleNotFoundError()

from ..base_class import BaseModel

class Yolov5TRT(BaseModel):
    def __init__():
        pass


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