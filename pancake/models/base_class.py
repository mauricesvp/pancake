from abc import ABC, abstractmethod
from typing import Type

import torch
import torch.backends.cudnn as cudnn

from pancake.utils.torch_utils import select_device


class BaseModel(ABC):
    _subclasses = {}

    def __init__(self, device: str):
        """
        This class acts as a base class for different models.

        :param device (torch.device): device to calculate on (cpu, gpu)
        :param weights (str): path to custom trained weights or name of the official pretrained yolo
        """
        self.model = None
        self._device = select_device(device)
        self._half = self._device.type != "cpu"  # half precision only supported on CUDA

        self._stride = None
        self._required_img_size = None

        cudnn.benchmark = True  # set True to speed up constant image size inference

    @classmethod
    def get_subclasses(cls):
        return dict(cls._subclasses)

    def __init_subclass__(cls):
        module_name = cls.__module__.replace("pancake.models.", "")
        if module_name.endswith("_class"):
            module_name = module_name[:-6]
        BaseModel._subclasses[module_name] = cls

    @abstractmethod
    def _init_infer(self, img_size):
        """
        Does one forward pass on the network for initialization on gpu
        :param img_size: padded, resized image size
        """
        assert img_size, "Your model needs to specify a specific image size "
        "for inference in class attribute '._required_img_size'"
        if self._device.type != "cpu":
            self.model(
                torch.zeros(1, 3, img_size, img_size)
                .to(self._device)
                .type_as(next(self.model.parameters()))
            )  # run once

    @abstractmethod
    def prep_image_infer(self, img) -> Type[torch.Tensor]:
        """
        Preprocesses images for inference (on device, expanded dim (,4), half precision (fp16), normalized)

        :param img: padded and resized image
        :return prep_img: preprocessed image
        """
        prep_img = torch.from_numpy(img).to(self._device)  # outsource on device
        prep_img = (
            prep_img.half() if self._half else prep_img.float()
        )  # uint8 to fp16/32
        prep_img /= 255.0  # 0 - 255 to 0.0 - 1.0 (normalize)

        if prep_img.ndimension() == 3:  # unsqueeze to conform model input format
            prep_img = prep_img.unsqueeze(0)

        return prep_img

    @abstractmethod
    def infer(self, img: Type[torch.Tensor]) -> Type[torch.Tensor]:
        """
        Infers on the given image.

        :param img (tensor): resized and padded image preprocessed for inference (meeting stride-multiple constraints),
                             4d tensor [x, R, G, B]
        :return list of detections, on (,6) tensor [xyxy, conf, cls]
        """
        assert (
            img.ndimension() == 4
        ), "Dimension of image array didn't match the required dimension (4)!"
        # Inference
        return self.model(img, augment=False)[0]
