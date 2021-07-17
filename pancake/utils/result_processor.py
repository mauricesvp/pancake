""" Holds the Pancake Result Processor """
from datetime import datetime
from typing import List

import cv2
import numpy as np
import torch

from .general import (
    check_imshow,
    resize_aspectratio,
    check_requirements,
)

from .plots import colors, plot_one_box


class ResultProcessor:
    def __init__(
        self,
        show_res: bool,
        save_res: bool,
        draw_det: bool,
        draw_tracks: bool,
        draw_track_hist: bool,
        track_hist_size: int,
        labels: List[str],
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
        """This class encapsulates all result processing procedures.

        Functionalities:
            - Draws detections, draws tracks, draws track history (tracked vehicle trajectories)
            - Visualizes the results, saves the results as image/video
            - Asynchronous approach, guaranteed throughput speedup!

        Args:
            show_res (bool): If resulting images should be visualized live
            save_res (bool): If resulting images should be saved
            draw_det (bool): If detection bbox' should be visualized
            draw_tracks (bool): If tracked object bbox' should be visualized
            draw_track_hist (bool): If track histories should be saved and visualized
            track_hist_size (int): Maximum size of the track hist container
            labels (List[str]): List of model specific class labels
            hide_labels (bool): If class labels and track ids should be visualized
            hide_conf (bool): If confidences should be visualized
            line_thickness (int): Drawn line thickness
            save_mode (str): Enum, "image" or "video"
            path (str): Parent target directory
            subdir (str): Target subdirectory (will be automatically incremented)
            exist_ok (bool): Save results in already existing dir (*path/subdir*),
                             do not increment automatically
            async_processing (bool): (non-blocking) Asynchronous result processing in a
                                     seperate slave process (significant speedup guaranteed!)
            async_queue_size (int): Queue size for results sent by parent process to subprocess
            async_put_blocked (bool): Blocks .process() for timeout sec until free slot
                                        available, if False skip the current frame without
                                        blocking
            async_put_timeout (float): Raise exception after timeout seconds waiting for spare slot
            debug (bool): Manual skipping when visualizing results
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

        if self._show_track_hist:
            """INIT THE TRACK HISTORY"""

            class TrackHistory:
                """Track History Wrapper
                - Store the latest tracking results
                - Assign each tracked ID its center positions (x, y)
                """

                def __init__(self, max_hist_len: int):
                    """ Track History Wrapper

                    Args:
                        max_hist_len (int): Maximum age of tracks matrix considered for \
                                            trajectory visualization
                    """
                    self.tracks = []
                    self.ids = {}
                    self._max_hist_len = max_hist_len

                def update(self, tracks: np.ndarray):
                    if len(self.tracks) > self._max_hist_len:
                        self.tracks = []
                        self.ids = {}

                    self.tracks.append(tracks)
                    curr_ids = []
                    for *_, x, y, id, _ in tracks:
                        if id not in self.ids.keys():
                            self.ids.update({id: [(x, y)]})
                        else:
                            self.ids[id].append((x, y))
                        curr_ids.append(id)

            self.track_history = TrackHistory(max_hist_len=track_hist_size)

        if self._save_res:
            """INITIALIZE THE SAVING PROCEDURE"""
            assert save_mode == "video" or save_mode == "image"
            from .general import increment_path
            from pathlib import Path

            self._mode = save_mode

            self._save_dir = increment_path(Path(path) / subdir, exist_ok, sep="_")
            self._save_dir.mkdir(parents=True, exist_ok=True)

            self.vid_path, self.vid_writer, self._fps = (
                None,
                None,
                kwargs["vid_fps"] if "vid_fps" in kwargs.keys() else None,
            )

        if self._async:
            """INITIALIZE THE ASYNCHRONOUS RESULT PROCESSING"""
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
        det: torch.Tensor,
        tracks: np.ndarray,
        im0: np.ndarray,
        vid_cap: cv2.VideoCapture = None,
    ):
        """Wraps the procedure for asynchronous and synchronous result processing.

        This method acts as the entrypoint for the results originating from the run script.

        Args:
            det (torch.Tensor): Detections on (,6) tensor [xyxy, conf, cls]
            tracks (np.array): Tracks on (,7) array [xyxy, center x, center y, id]
            im0 (np.array): Image in BGR (,3) [c, w, h]
            vid_cap (cv2.VideoCapture, optional): _Deprecated_ cv2.VideoCapture object. Defaults to None.
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
        det: torch.Tensor,
        tracks: np.ndarray,
        im0: np.ndarray,
        vid_cap: cv2.VideoCapture = None,
    ):
        """Wraps the actual result processing procedures.
        Takes the provided results from a detector and tracker in order to visualize
        them according to user config. Subsequently, visualizes and/or stores the
        enriched image/video.

        Args:
            det (torch.Tensor): Detections on (,6) tensor [xyxy, conf, cls]
            tracks (np.array): Tracks on (,7) array [xyxy, center x, center y, id]
            im0 (np.array): Image in BGR (,3) [c, w, h]
            vid_cap (cv2.VideoCapture, optional): _Deprecated_ cv2.VideoCapture object. Defaults to None.
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
        """The slave process loop"""
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

    def draw_detec_boxes(self, det: torch.Tensor, im0: np.ndarray) -> np.ndarray:
        """Draws bounding boxes, class labels and confidences according to a
        detection matix on the provided image.

        Args:
            det (torch.Tensor): Detections on (,6) tensor [xyxy, conf, cls]
            im0 (np.ndarray): Image in BGR [3, w, h]

        Returns:
            np.ndarray: Image enriched with detection boxes
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

    def draw_track_boxes(self, tracks: np.ndarray, im0: np.ndarray) -> np.ndarray:
        """Draws bounding boxes, tracking ids according to 'tracks' matix on the provided image.

        Args:
            tracks (np.ndarray): Tracks on (,7) array [xyxy, center x, center y, id, cls]
            im0 (np.ndarray): Image in BGR [3, w, h]

        Returns:
            np.ndarray: Image enriched with tracked object boxes
        """
        for *xyxy, _, _, id, _ in tracks:
            id = None if self._hide_labels else str(id)
            plot_one_box(
                xyxy,
                im0,
                label=id,
                color=colors(int(id), True),
                line_thickness=self._line_thickness,
            )
        return im0

    def draw_track_hist(self, tracks: np.ndarray, im0: np.ndarray) -> np.ndarray:
        """Draws a line for each tracked ID according to the stored history.

        Args:
            tracks (np.ndarray): Tracks on (,7) array [xyxy, center x, center y, id, cls]
            im0 (np.ndarray): Image in BGR [3, w, h]

        Returns:
            np.ndarray: Image enriched with tracked object trajectories
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

    def save_img(self, im0: np.ndarray):
        """Saves an image with current timestamp as name.

        Args:
            im0 (np.ndarray): Image in BGR [3, w, h]
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # image named after current timestamp
        save_path = str(self._save_dir / now)
        save_path += ".jpg"

        cv2.imwrite(save_path, im0)

    def save_vid(self, im0: np.ndarray, vid_cap: cv2.VideoCapture = None):
        """Appends an image to the current videopointer.

        Args:
            im0 (np.ndarray): Image in BGR [3, w, h]
            vid_cap (cv2.VideoCapture, optional): _Deprecated_ cv2.VideoCapture object. Defaults to None.
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

    def visualize(self, im0: np.ndarray):
        """Visualizes the passed image.

        Args:
            im0 (np.ndarray): Image in BGR [3, w, h]
        """
        cv2.namedWindow("Pancake", cv2.WINDOW_NORMAL)
        cv2.imshow("Pancake", im0)
        cv2.waitKey(0 if self._debug else 1)

    def reset_vid_writer(self):
        self.vid_writer.release()
        self.vid_path, self.vid_writer = None, None


def setup_result_processor(config: dict, labels: list) -> ResultProcessor:
    """Helper function to retrieve the configs and parse it to the
        ResultProcessor class initialization.

    Args:
        config (dict): Dictionary containing the configurations
        labels (list): Detector specific object labels

    Returns:
        ResultProcessor: An instance of the ResultProcessor
    """
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
