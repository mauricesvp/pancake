import os
import sys
from datetime import datetime
from typing import Type, List, Union


import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ..models.base_class import BaseModel
from .datasets import LoadStreams, LoadImages, LoadWebcam, LoadImageDirs
from .general import (
    check_img_size, scale_coords, check_imshow, resize_aspectratio)
from .plots import colors, plot_one_box
from .torch_utils import time_synchronized


def load_data(
    source: str
) -> Union[LoadStreams, LoadImages, LoadImageDirs]:
    """
    :param source (str): data source (webcam, image, video, directory, glob, youtube video, HTTP stream)
    """
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
                LoadStreams(source),
                True,
            )
        elif type(source) is list:
            return (
                LoadImageDirs(source),
                False,
            )
        else:
            return (
                LoadImages(source),
                False,
            )

def draw_boxes(
    show_det: bool,
    show_tracks: bool,
    im0: Type[np.array],
    hide_labels: bool,
    hide_conf: bool,
    line_thickness: int,
    labels: List = None,
    det: Type[torch.Tensor] = None,
    tracks: Type[np.ndarray] = None
) -> Type[np.array]:
    """
    :param show_det (bool): if detection bbox' should be visualized
    :param show_tracks (bool): if tracked object bbox' should be visualized
    :param det (tensor): detections on (,6) tensor [xyxy, conf, cls]
    :param tracks (np.ndarray): track ids on (,7) array [xyxy, center x, center y, id]
    :param im0 (array): original image
    :param labels (list): list of model specific class labels
    :param hide_labels (bool): if labels should be visualized
    :param hide_conf (bool): if confidences should be visualized
    :param line_thickness (int): line thickness

    :return (ndarray) image enriched with provided boxes
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
    return im0
    

def visualize(
    im0: Type[np.array],
    debug: bool=False
) -> None:
    """
    :param im0 (array): original image
    :param debug (bool): enables debug stepping
    """
    cv2.namedWindow("Pancake", cv2.WINDOW_NORMAL)
    cv2.imshow("Pancake", im0)
    cv2.waitKey(0 if debug else 1)

def save(
    im0: Type[np.array],
    vid_cap: Type[cv2.VideoCapture] = None,
    vid_fps: int = 30,
    mode: str = "image",
    path: str = "../pancake_results"
) -> None:
    """
    :param im0 (array): image to save
    :param vid_cap (cv2.VideoCapture): cv2.VideoCapture object
    :param vid_fps (int): fixed video frame rate when in video mode
    :param mode (str): "image" or "video"
    :param path (str): target directory
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # image named after timestamp
    save_path = str(path / now)

    if mode.lower() == "image":
        save_path += ".jpg"
        cv2.imwrite(save_path, im0)
    else:  # 'video' or 'stream'
        if not "vid_path" in globals() or not "vid_writer" in globals():
            globals()["vid_path"],  globals()["vid_writer"] = None, None

        if im0.shape[1] > 3200:
            im0 = resize_aspectratio(im0, width=3200)

        if not vid_path:  # new video
            globals()["vid_path"] = save_path

            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer

            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # images, stream
                fps, w, h = vid_fps, im0.shape[1], im0.shape[0]
                save_path += '.avi'

            globals()["vid_writer"] = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        vid_writer.write(im0)


def fix_path(path: Union[str, list]) -> str:
    """Adjust relative path."""
    if type(path) is list:
        return list(map(lambda p: fix_path(p), path))
    return os.path.join(os.path.dirname(__file__), "..", path)

