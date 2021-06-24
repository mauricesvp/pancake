import os
import sys
from datetime import datetime
from typing import Type, List, Union

import cv2
import numpy as np
import torch

from pancake.models.base_class import BaseModel

from .datasets import LoadStreams, LoadImages, LoadWebcam, LoadImageDirs
from .general import (
    check_img_size,
    scale_coords,
    check_imshow,
    resize_aspectratio,
    check_requirements,
)
from .plots import colors, plot_one_box
from .torch_utils import time_synchronized


def load_data(source: str) -> Union[LoadStreams, LoadImages, LoadImageDirs]:
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


def setup_result_processor(config: dict, labels: list):
    return ResultProcessor(
        show_res=config.VIEW_RES,
        save_res=config.SAVE_RES,
        draw_det=config.DRAW_DET,
        draw_tracks=config.DRAW_TRACKS,
        draw_track_hist=config.DRAW_TRACK_HIST,
        track_hist_size=config.MAX_TRACK_HIST_LEN,
        labels=labels,
        hide_labels=config.HIDE_LABELS,
        hide_conf=config.HIDE_CONF,
        line_thickness=config.LINE_THICKNESS,
        save_mode=config.MODE,
        path=config.PATH,
        subdir=config.SUBDIR,
        exist_ok=config.EXIST_OK,
        vid_fps=config.VID_FPS,
        async_processing=config.ASYNC_PROC,
        async_queue_size=config.Q_SIZE,
        async_put_blocked=config.PUT_BLOCKED,
        async_put_timeout=config.PUT_TIMEOUT,
        debug=config.DEBUG,
    )


class ResultProcessor:
    def __init__(
        self,
        show_res: bool,
        save_res: bool,
        draw_det: bool,
        draw_tracks: bool,
        draw_track_hist: bool,
        track_hist_size: int,
        labels: list,
        hide_labels: bool,
        hide_conf: bool,
        line_thickness: int,
        save_mode: str,
        path: str,
        subdir: str,
        exist_ok: bool,
        async_processing: bool,
        async_queue_size: int,
        async_put_blocked: bool,
        async_put_timeout: float,
        debug: bool,
        *args,
        **kwargs,
    ):
        """
        :param show_res (bool): if resulting images should be visualized live
        :param save_res (bool): if resulting images should be saved
        :param draw_det (bool): if detection bbox' should be visualized
        :param draw_tracks (bool): if tracked object bbox' should be visualized
        :param draw_track_hist (bool): if track histories should be saved and visualized
        :param track_hist_size (int): maximum size of the track hist container
        :param labels (list): list of model specific class labels
        :param hide_labels (bool): if labels should be visualized
        :param hide_conf (bool): if confidences should be visualized
        :param line_thickness (int): line thickness
        :param save_mode (str): "image" or "video"
        :param path (str): parent target directory
        :param subdir (str): target subdirectory
        :param exist_ok (str): save in results in already existing dir
                               do not increment automatically
        :param async_processing (str): (non-blocking) asynchronous result processing in a
                                       seperate slave process
        :param async_queue_size (int): queue size for results sent by .process() to subprocess
        :param async_put_blocked (bool): blocks .process() for timeout sec until free slot
                                         available, if False skip the current frame without
                                         blocking
        :param async_put_timeout (float): raise exception after timeout s waiting for free slot
        :param debug (bool): manual skipping when visualizing results
        """
        from pancake.run import setup_logger

        self.l = setup_logger(__name__)

        # GENERAL
        self._show_res, self._save_res, self._debug, self._async = (
            False,
            save_res,
            debug,
            async_processing,
        )
        if show_res:
            self._show_res = True if check_imshow() else False

        # DRAW OPTIONS
        self._show_det, self._show_tracks, self._show_track_hist = (
            draw_det,
            draw_tracks,
            draw_track_hist,
        )
        # DRAW DETAILS
        self._hide_labels, self._hide_conf = hide_labels, hide_conf
        self._line_thickness = line_thickness
        # CLASS LABELS
        self._labels = labels

        # NEITHER SHOW RES OR SAVE RES IS ENABLED
        if not self._show_res and not self._save_res:
            self.l.info("No result processing procedure will be taking place")

            def nop(*args, **kwargs):
                return

            self.process = nop
            self.kill_worker = nop
            return

        # INITIALIZE TRACK HISTORY
        if self._show_track_hist:

            class TrackHistory:
                """Track History Wrapper
                - store the latest tracking results
                - assign each tracked ID its center positions (x, y)
                """

                def __init__(self, max_hist_len: int):
                    self.tracks = []
                    self.ids = {}
                    self._max_hist_len = max_hist_len

                def update(self, tracks):
                    if len(self.tracks) > self._max_hist_len:
                        self.tracks = []
                        self.ids = {}

                    self.tracks.append(tracks)
                    curr_ids = []
                    for *_, x, y, id in tracks:
                        if id not in self.ids.keys():
                            self.ids.update({id: [(x, y)]})
                        else:
                            self.ids[id].append((x, y))
                        curr_ids.append(id)

            self.track_history = TrackHistory(max_hist_len=track_hist_size)

        # INIT SAVING
        if self._save_res:
            assert save_mode == "video" or save_mode == "image"
            from .general import increment_path
            from pathlib import Path

            self._mode = save_mode

            self._save_dir = increment_path(Path(path) / subdir, exist_ok)
            self._save_dir.mkdir(parents=True, exist_ok=True)

            self.vid_path, self.vid_writer, self._fps = (
                None,
                None,
                kwargs["vid_fps"] if "vid_fps" in kwargs.keys() else None,
            )

        # INIT ASYNC RES PROCESSING
        if self._async:
            import multiprocessing

            check_requirements(["pathos"])
            from pathos.helpers import mp

            assert (
                not self._show_res and self._async
            ), "Results can't be visualized from slave process, disable 'VIEW_IMG' or 'ASYNC_PROC'!"
            assert (
                multiprocessing.cpu_count() > 1
            ), "Only 1 CPU core available, might not be able to leverage multiprocessing module!"

            self._queue_size = async_queue_size
            self._put_blocked = async_put_blocked
            self._put_timeout = async_put_timeout

            # Queue to put and pull results
            self.queue = mp.Queue(maxsize=self._queue_size)

            # init and start worker process
            self.l.info("Starting slave process to work on the results")
            self.worker_process = mp.Process(target=self.async_update_worker, args=())
            self.worker_process.start()

    def process(
        self,
        det: Type[torch.Tensor],
        tracks: Type[np.array],
        im0: Type[np.array],
        vid_cap: Type[cv2.VideoCapture] = None,
    ):
        """
        Wraps the procedure for asynchronous and synchronous result processing.

        :param det (tensor): detections on (,6) tensor [xyxy, conf, cls]
        :param tracks (np.ndarray): track ids on (,7) array [xyxy, center x, center y, id]
        :param im0 (array): image in BGR (,3) [3, px, px]
        :param vid_cap (cv2.VideoCapture): cv2.VideoCapture object
        """
        if self._async:
            assert self.worker_process.is_alive(), "Worker process died!"

            if round(self.queue.qsize() / self._queue_size, 1) == 0.9:
                self.l.warn(
                    "Queue size capacity almost full.. "
                    f"({(self.queue.qsize()/self._queue_size) * 100:.2f})"
                )

            if not self._put_blocked and self.queue.full():
                return

            # self.queue.put([det, tracks, im0, vid_cap])
            self.queue.put(
                [det, tracks, im0], block=self._put_blocked, timeout=self._put_timeout
            )
        else:
            # self.update(det, tracks, im0, vid_cap)
            self.update(det, tracks, im0)

    def update(
        self,
        det: Type[torch.Tensor],
        tracks: Type[np.array],
        im0: Type[np.array],
        vid_cap: Type[cv2.VideoCapture] = None,
    ):
        """
        Takes the provided results from a detector and tracker in order to visualize
        them according to user config. Subsequently, visualizes and/or stores the
        enriched image/video.

        :param det (tensor): detections on (,6) tensor [xyxy, conf, cls]
        :param tracks (np.ndarray): track ids on (,7) array [xyxy, center x, center y, id]
        :param im0 (array): image in BGR (,3) [3, px, px]
        :param vid_cap (cv2.VideoCapture): cv2.VideoCapture object
        """
        if self._show_det:
            im0 = self.draw_detec_boxes(det, im0)
        if self._show_tracks:
            im0 = self.draw_track_boxes(tracks, im0)
        if self._show_track_hist:
            im0 = self.draw_track_hist(tracks, im0)

        if self._show_res:
            self.visualize(im0)

        if self._save_res:
            if self._mode == "image":
                self.save_img(im0)
            else:
                self.save_vid(im0)

    def async_update_worker(self):
        """
        Main loop of the worker process
        """
        assert self.queue, "Queue is not initialized!"
        import traceback

        try:
            while 1:
                try:
                    data = self.queue.get()
                except:
                    self.reset_vid_writer()
                    self.queue.close()
                    break

                # deserialize data, data: list, [detections, tracks, image, vid cap]
                # det, tracks, im0, vid_cap = data[0], data[1], data[2], data[3]
                det, tracks, im0 = data[0], data[1], data[2]

                self.update(det, tracks, im0)
        except:
            self.l.fatal("Exception in subprocess occured!")
            traceback.print_exc()

    def kill_worker(self):
        """
        Procedure for cleanly closing the communication pipes and terminating
        the worker process.
        """
        self.queue.close()
        self.worker_process.terminate()

    def draw_detec_boxes(self, det: Type[torch.Tensor], im0: Type[np.array]):
        """
        Draws bounding boxes, class labels and confidences according to a
        detection matix on the provided image.

        :param det (tensor): detections on (,6) tensor [xyxy, conf, cls]
        :param im0 (ndarray): image in BGR [3, px, px]
        """
        for *xyxy, conf, cls in reversed(det):
            # Add bbox to image
            c = int(cls)  # integer class
            label = (
                None
                if self._hide_labels
                else (
                    self._labels[c]
                    if self._hide_conf
                    else f"{self._labels[c]} {conf:.2f}"
                )
            )

            plot_one_box(
                xyxy,
                im0,
                label=label,
                color=colors(c, True),
                line_thickness=self._line_thickness,
            )
        return im0

    def draw_track_boxes(self, tracks: Type[np.array], im0: Type[np.array]):
        """
        Draws bounding boxes, tracking ids according to a tracks matix on the provided image.

        :param tracks (np.ndarray): track ids on (,7) array [xyxy, center x, center y, id]
        :param im0 (array): image in BGR [3, px, px]
        """
        for *xyxy, _, _, id in tracks:
            id = None if self._hide_labels else str(id)
            plot_one_box(
                xyxy,
                im0,
                label=id,
                color=colors(int(id), True),
                line_thickness=self._line_thickness,
            )
        return im0

    def draw_track_hist(self, tracks: Type[np.array], im0: Type[np.array]):
        """
        Draws a line for each tracked ID according to the stored history.

        :param tracks (np.ndarray): track ids on (,7) array [xyxy, center x, center y, id]
        :param im0 (array): image in BGR [3, px, px]
        """
        assert self.track_history, "No track history object initialized!"
        self.track_history.update(tracks)

        for id in self.track_history.ids:
            points = self.track_history.ids[id]
            x0, y0 = points[0]
            for x, y in points[1:]:
                cv2.line(im0, (x0, y0), (x, y), colors(id, True), self._line_thickness)
                x0, y0 = x, y
        return im0

    def save_img(self, im0: Type[np.array]):
        """
        :param im0 (ndarray): image in BGR [3, px, px]
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # image named after current timestamp
        save_path = str(self._save_dir / now)
        save_path += ".jpg"

        cv2.imwrite(save_path, im0)

    def save_vid(self, im0: Type[np.array], vid_cap: Type[cv2.VideoCapture] = None):
        """
        :param im0 (ndarray): image in BGR [3, px, px]
        :param vid_cap (cv2.VideoCapture): cv2.VideoCapture object
        """
        # resize frame if its too large for .avi
        if im0.shape[1] > 3200:
            im0 = resize_aspectratio(im0, width=3200)

        if not self.vid_path:  # new video
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.vid_path = str(self._save_dir / now)

            if isinstance(self.vid_writer, cv2.VideoWriter):
                self.vid_writer.release()  # release previous video writer

            # take w, h from input video
            # if vid_cap:  # video
            #     fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # else:  # images, stream

            # take user defined fps
            fps, w, h = self._fps, im0.shape[1], im0.shape[0]
            self.vid_path += ".avi"

            self.vid_writer = cv2.VideoWriter(
                self.vid_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h)
            )

        self.vid_writer.write(im0)

    def visualize(self, im0: Type[np.array]):
        """
        :param im0 (ndarray): image in BGR [3, px, px]
        """
        cv2.namedWindow("Pancake", cv2.WINDOW_NORMAL)
        cv2.imshow("Pancake", im0)
        cv2.waitKey(0 if self._debug else 1)

    def reset_vid_writer(self):
        self.vid_writer.release()
        self.vid_path, self.vid_writer = None, None


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
