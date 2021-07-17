import os
from datetime import datetime
from typing import Type, List, Union

import cv2
import numpy as np
import torch

from .datasets import LoadStreams, LoadImages, LoadWebcam, LoadImageDirs
from .general import resize_aspectratio
from .plots import colors, plot_one_box


def load_data(
    source: Union[List[str], str]
) -> Union[LoadStreams, LoadImages, LoadImageDirs]:
    """ Sets up the data loader

    Description:
        Depending on the source, returns a class responsible for providing image frames to the
        Pancake main loop.
    
    Source Types
    ------------
        - Single image: source contains path to single image (e.g. "../samples/r45/1c/1621796022.9767.jpg")
        - Single video: source contains path to single video (e.g. "../samples/output.avi")
        - Sequence of images: source contains path to a directory (e.g. "../samples/r45/1c")
        - Multiple directories with images/videos: source is a list containing paths to different directories \
            (e.g. List["../samples/r45/1l", "../samples/r45/1c", "../samples/r45/1r"])
        - Multiple web streams: source is a .txt file containing different web adresses
            (stream could be from e.g. YouTube, webcam, Twitch, ...)

    Args:
        source (Union[List[str], str]): Data source

    Returns:
        Union[LoadStreams, LoadImages, LoadImageDirs]: A data loader object
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
    tracks: Type[np.ndarray] = [],
    track_history: Type[list] = [],
    show_track_history: bool = False,
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
    if show_track_history:
        id_hist = {}
        init = True
        for state in track_history:
            for *_, x, y, id in state:
                if init:
                    id_hist.update({id: [(x, y)]})
                elif id not in id_hist:
                    continue
                else:
                    id_hist[id].append((x, y))
            init = False
        for id in id_hist:
            points = id_hist[id]
            x0, y0 = points.pop(0)
            for x, y in points:
                cv2.line(im0, (x0, y0), (x, y), colors(id, True), 5)
                x0, y0 = x, y
    return im0


def visualize(im0: Type[np.array], debug: bool = False) -> None:
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
    path: str = "../pancake_results",
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
            globals()["vid_path"], globals()["vid_writer"] = None, None

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
                save_path += ".avi"

            globals()["vid_writer"] = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h)
            )
        vid_writer.write(im0)


def fix_path(path: Union[str, list]) -> str:
    """Adjust relative path."""
    if type(path) is list:
        return list(map(lambda p: fix_path(p), path))
    return os.path.join(os.path.dirname(__file__), "..", path)
