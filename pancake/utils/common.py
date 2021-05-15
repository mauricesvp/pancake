import cv2
import sys
import torch
import torch.backends.cudnn as cudnn
from typing import Type, List, Union

from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_imshow
from utils.plots import colors, plot_one_box

def load_data(source: str, 
              model
              ) -> Union[LoadStreams, LoadImages]:
    """
    :param source (str): data source (webcam, image, video, directory, glob, youtube video, HTTP stream)
    :param model (BaseModel): model wrapper
    :param img_size (int): inference size (pixels)
    """
    assert (model._required_img_size
    ), "Your model needs to specify a model specific image size " 
    "in class attribute '._required_img_size'"

    is_webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )
    if is_webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        return LoadStreams(source, img_size=model._required_img_size, stride=model._stride), True
    else:
        return LoadImages(source, img_size=model._required_img_size, stride=model._stride), False

def visualize(det: Type[torch.Tensor],
              p: str,
              im0, 
              labels: List,
              hide_labels: bool,
              hide_conf: bool,
              line_thickness: int
              ) -> None:
    """
    :param det (tensor): detections on (,6) tensor [xyxy, conf, cls] 
    :param p (str): path of image
    :param im0s (array): original image
    :param labels (list): list of model specific class labels
    :param hide_labels (bool): if labels should be visualized
    :param hide_conf (bool): if confidences should be visualized
    :param line_thickness (int): line thickness 
    """
    # Draw boxes
    for *xyxy, conf, cls in reversed(det):
        # Add bbox to image
        c = int(cls)  # integer class
        label = None if hide_labels else (labels[c] if hide_conf else f'{labels[c]} {conf:.2f}')

        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
    
    cv2.imshow(str(p), im0)
    cv2.waitKey(1)  # 1 millisecond