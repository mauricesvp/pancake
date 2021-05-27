import logging

import cv2
import numpy as np
import torch

from . import detector as det
from . import tracker as tr
from .detector import backends as be

from .config import pancake_config
from .logger import setup_logger
from .utils.common import fix_path, load_data, visualize
from .utils.parser import get_config

l = setup_logger(__name__)


def setup_detector(config):
    name = config.DETECTOR.NAME
    params = getattr(config.DETECTOR, name.upper())

    device_cfg = config.DEVICE
    device = "CPU"

    if type(device_cfg) is str and torch.cuda.is_available():
        if device_cfg.upper() == "GPU":
            device = "0"
        elif device_cfg.isdigit():
            device = device_cfg

    DETECTOR = det.DETECTOR_REGISTRY[name](params, device=device)
    return DETECTOR


def setup_backend(config, detector):
    name = config.DETECTOR.BACKEND

    ROI = config.DATA.ROI
    return be.BACKEND_REGISTRY[name](detector, roi=ROI)


def setup_tracker(config):
    name = config.TRACKER.NAME
    params = getattr(config.TRACKER, name.upper())
    tracker_cfg = get_config(config_file=fix_path(params.TRACKER_CFG_PATH))

    device_cfg = config.DEVICE
    device = "CPU"

    if type(device_cfg) is str and torch.cuda.is_available():
        if device_cfg.upper() == "GPU":
            device = "0"
        elif device_cfg.isdigit():
            device = device_cfg

    TRACKER = tr.TRACKER_REGISTRY[name](tracker_cfg, device=device)
    return TRACKER


def setup_logging(config):
    try:
        log_level = getattr(logging, config.LOGGING.LEVEL)
        l.setLevel(log_level)
    except:
        l.setLevel(logging.INFO)
    l.debug(f"Log level set to {log_level}.")


def main():
    l.debug("Starting pancake.")

    config = pancake_config()

    setup_logging(config.PANCAKE)

    # Detector setup
    DETECTOR = setup_detector(config.PANCAKE)
    BACKEND = setup_backend(config.PANCAKE, DETECTOR)

    # Tracker setup
    TRACKER = setup_tracker(config.PANCAKE)

    # Input data setup
    source = config.PANCAKE.DATA.SOURCE
    source_path = fix_path(source)

    DATA, is_webcam = load_data(source_path)

    iteration = 0
    for path, img, im0s, vid_cap in DATA:
        l.debug(f"Iteration {iteration}")
        iteration += 1

        detections = BACKEND.detect(im0s)
        stitched = cv2.hconcat([im0s[0], im0s[1], im0s[2]])
        tracks = TRACKER.update(detections, stitched)

        if config.PANCAKE.VISUALIZATION.VIEW_IMG:
            hide_labels = config.PANCAKE.VISUALIZATION.HIDE_LABELS
            hide_conf = config.PANCAKE.VISUALIZATION.HIDE_CONF
            line_thickness = config.PANCAKE.VISUALIZATION.LINE_THICKNESS
            visualize(
                show_det=True,
                show_tracks=True,
                det=detections,
                tracks=tracks,
                # im0=im0,
                im0=stitched,
                labels=DETECTOR.model.names,
                hide_labels=hide_labels,
                hide_conf=hide_conf,
                line_thickness=line_thickness,
                debug=False,  # Set to True to enable manual stepping
            )


if __name__ == "__main__":
    main()
