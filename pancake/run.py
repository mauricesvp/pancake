import argparse
import logging

import cv2
import numpy as np
import torch
from pathlib import Path

from . import detector as det
from . import tracker as tr
from .detector import backends as be

from .config import pancake_config
from .logger import setup_logger
from .utils.common import fix_path, load_data, ResultProcessor

l = setup_logger(__name__)


def setup_logging(config):
    try:
        log_level = getattr(logging, config.LOGGING.LEVEL)
        l.setLevel(log_level)
    except:
        l.setLevel(logging.INFO)
    l.debug(f"Log level set to {log_level}.")


def main(cfg_path: str = None, n: int = 0):
    """

    :param cfg_path (str): Alternative config path
    :param n (int): Maximum number of iterations (0 means infinite)
    """
    l.debug("Starting pancake.")

    config = pancake_config(cfg_path)

    setup_logging(config.PANCAKE)

    # Detector setup
    DETECTOR = det.setup_detector(config.PANCAKE)
    BACKEND = be.setup_backend(config.PANCAKE, DETECTOR)

    # Tracker setup
    TRACKER = tr.setup_tracker(config.PANCAKE)

    # Input data setup
    source = config.PANCAKE.DATA.SOURCE
    source_path = fix_path(source)

    DATA, _ = load_data(source_path)

    # Visualization and save configs
    res_cfg = config.PANCAKE.RESULT_PROCESSING

    RESULT_PROC = ResultProcessor(
        show_res=res_cfg.VIEW_RES,
        save_res=res_cfg.SAVE_RES,
        draw_det=res_cfg.DRAW_DET,
        draw_tracks=res_cfg.DRAW_TRACKS,
        draw_track_hist=res_cfg.DRAW_TRACK_HIST,
        track_hist_size=res_cfg.MAX_TRACK_HIST_LEN,
        labels=DETECTOR.model.names,
        hide_labels=res_cfg.HIDE_LABELS,
        hide_conf=res_cfg.HIDE_CONF,
        line_thickness=res_cfg.LINE_THICKNESS,
        save_mode=res_cfg.MODE,
        path=res_cfg.PATH,
        subdir=res_cfg.SUBDIR,
        exist_ok=res_cfg.EXIST_OK,
        vid_fps=res_cfg.VID_FPS,
        async_processing=res_cfg.ASYNC_PROC,
        debug=res_cfg.DEBUG,
    )

    iteration = 0
    for path, im0s, vid_cap in DATA:
        l.debug(f"Iteration {iteration}")
        iteration += 1

        detections, frame = BACKEND.detect(im0s)

        tracks = TRACKER.update(detections, frame)

        RESULT_PROC.process(detections, tracks, frame, vid_cap)

        if n and iteration >= n:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", nargs="?", type=str, default=None, help="pancake config path"
    )
    args = args = parser.parse_args()

    main(args.cfg)
