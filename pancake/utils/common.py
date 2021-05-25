import cv2
import sys
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from typing import Type, List, Union

from ..models.base_class import BaseModel
from .datasets import LoadStreams, LoadImages, LoadWebcam, LoadImageDirs
from .general import check_img_size, scale_coords, check_imshow
from .plots import colors, plot_one_box
from .torch_utils import time_synchronized


def load_data(source: str, model: Type[BaseModel]) -> Union[LoadStreams, LoadImages]:
    """
    :param source (str): data source (webcam, image, video, directory, glob, youtube video, HTTP stream)
    :param model (BaseModel): model wrapper
    :param img_size (int): inference size (pixels)
    """
    assert (
        model._required_img_size
    ), "Your model needs to specify a model specific image size "
    "in class attribute '._required_img_size'"

    try:
        is_webcam = (
            source.isnumeric()
            or source.endswith(".txt")
            or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        )
    except AttributeError:
        is_webcam = False

    finally:
        if is_webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            return (
                LoadStreams(
                    source, img_size=model._required_img_size, stride=model._stride
                ),
                True,
            )
        elif type(source) is list:
            return (
                LoadImageDirs(
                    source, img_size=model._required_img_size, stride=model._stride
                ),
                False,
            )
        else:
            return (
                LoadImages(
                    source, img_size=model._required_img_size, stride=model._stride
                ),
                False,
            )


def visualize(
    show_det: bool,
    show_tracks: bool,
    det: Type[torch.Tensor],
    tracks: Type[np.ndarray],
    p: str,
    im0,
    labels: List,
    hide_labels: bool,
    hide_conf: bool,
    line_thickness: int,
    debug: bool = False,
) -> None:
    """
    :param show_det (bool): if detection bbox' should be visualized
    :param show_tracks (bool): if tracked object bbox' should be visualized
    :param det (tensor): detections on (,6) tensor [xyxy, conf, cls]
    :param tracks (np.ndarray): track ids on (,7) array [xyxy, center x, center y, id]
    :param p (str): path of image
    :param im0s (array): original image
    :param labels (list): list of model specific class labels
    :param hide_labels (bool): if labels should be visualized
    :param hide_conf (bool): if confidences should be visualized
    :param line_thickness (int): line thickness
    :param debug (bool): enables debug stepping
    """
    # Draw boxes
    if show_det:
        for *xyxy, conf, cls in reversed(det):
            # Add bbox to image
            c = int(cls)  # integer class
            label = (
                None
                if hide_labels
                else (labels[c] if hide_conf else f"{labels[c]} {conf:.2f}")
            )

            plot_one_box(
                xyxy,
                im0,
                label=label,
                color=colors(c, True),
                line_thickness=line_thickness,
            )

    if show_tracks:
        for *xyxy, _, _, id in tracks:
            plot_one_box(
                xyxy,
                im0,
                label=str(id),
                color=colors(int(id), True),
                line_thickness=line_thickness,
            )

    im0 = cv2.resize(im0, (1080, 640))
    cv2.imshow("Pancake", im0)
    cv2.waitKey(0 if debug else 1)
