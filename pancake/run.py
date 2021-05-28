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
from .utils.common import fix_path, load_data, draw_boxes, visualize, save
from .utils.general import increment_path

l = setup_logger(__name__)


def setup_logging(config):
    try:
        log_level = getattr(logging, config.LOGGING.LEVEL)
        l.setLevel(log_level)
    except:
        l.setLevel(logging.INFO)
    l.debug(f"Log level set to {log_level}.")


def main(cfg_path: str=None):
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

    DATA, is_webcam = load_data(source_path)

    # Visualization and save configs
    vis_cfg = config.PANCAKE.VISUALIZATION
    save_cfg = config.PANCAKE.SAVE_RESULT

    if save_cfg.SAVE_RES:
        save_dir = increment_path(
            Path(save_cfg.PATH) / save_cfg.SUBDIR, exist_ok=save_cfg.EXIST_OK
        )  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)

    iteration = 0
    for path, im0s, vid_cap in DATA:
        l.debug(f"Iteration {iteration}")
        iteration += 1

        detections = BACKEND.detect(im0s)

        if not type(im0s) is list:
            frame = im0s
        else:
            frame = cv2.hconcat([*im0s])
        tracks = TRACKER.update(detections, frame)

        if vis_cfg.VIEW_IMG or save_cfg.SAVE_RES:
            frame = draw_boxes(
                show_det=vis_cfg.SHOW_DET,
                show_tracks=vis_cfg.SHOW_TRACKS,
                det=detections,
                tracks=tracks,
                im0=frame,
                labels=DETECTOR.model.names,
                hide_labels=vis_cfg.HIDE_LABELS,
                hide_conf=vis_cfg.HIDE_CONF,
                line_thickness=vis_cfg.LINE_THICKNESS,
            )

            if vis_cfg.VIEW_IMG:
                visualize(im0=frame, debug=vis_cfg.DEBUG)

            if save_cfg.SAVE_RES:
                save(
                    im0=frame,
                    vid_cap=vid_cap,
                    vid_fps=save_cfg.VID_FPS,
                    mode=save_cfg.MODE,
                    path=save_dir,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='?', type=str, default='pancake/pancake.yaml', help='pancake config path')
    
    main()
