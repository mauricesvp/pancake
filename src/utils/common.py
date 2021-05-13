import cv2
import sys
import torch
import torch.backends.cudnn as cudnn
from typing import Type

sys.path.append('../models')
from models.base_class import BaseModel
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, scale_coords, check_imshow
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

def load_data(source: str, 
              model: Type[BaseModel],
              img_size: int, 
              is_webcam: bool):
    """
    :param source (str): data source (webcam, image, video, directory, glob, youtube video, HTTP stream)
    :param model (Model Wrapper): model wrapper
    :param img_size (int): inference size (pixels)
    :param is_webcam (bool): if data is sourced from webcam
    """
    if is_webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        return LoadStreams(source, img_size=img_size, stride=model._stride)
    else:
        return LoadImages(source, img_size=img_size, stride=model._stride)

def visualize(det: torch.Tensor,
              p: str,
              im0, 
              labels: list,
              hide_labels: bool,
              hide_conf: bool,
              line_thickness: int):
    """
    :param det (tensor): detections on (,6) tensor [xyxy, conf, cls] 
    :param p (str): path of image
    :param im0s (array): original image
    :param labels (list): list of model specific class labels
    """
    # Draw boxes
    for *xyxy, conf, cls in reversed(det):
        # Add bbox to image
        c = int(cls)  # integer class
        label = None if hide_labels else (labels[c] if hide_conf else f'{labels[c]} {conf:.2f}')

        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
    
    cv2.imshow(str(p), im0)
    cv2.waitKey(1)  # 1 millisecond