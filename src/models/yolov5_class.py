""" 
    encapsulates yolov5 functionalities (loading, inference)
"""
import torch

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device


class Yolov5_Model():
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
        self._device = select_device(device)
        self._half = self._device.type != 'cpu'  # half precision only supported on CUDA

        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._classes = classes
        self._agnostic_nms = agnostic_nms

        # load model
        self.model = attempt_load(weights, map_location=self._device)
        if self._half:
            self.model.half()  # to FP16
        
        self._stride = int(self.model.stride.max())  # model stride
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        print(f'Class names: {names}')

    def infer(self, img):
        """
        :param img: image
        :return list of detections, on (,6) tensor [xyxy, conf, cls]
        """
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        #t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms)
        #t2 = time_synchronized()

        return pred

    def _init_infer(self, img_size):
        if self._device.type != 'cpu':
            self.model(torch.zeros(1, 3, img_size, img_size).to(self._device).type_as(next(self.model.parameters())))  # run once
    

        
