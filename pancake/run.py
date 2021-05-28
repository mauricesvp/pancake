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

l = setup_logger(__name__)


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
    DETECTOR = det.setup_detector(config.PANCAKE)
    BACKEND = be.setup_backend(config.PANCAKE, DETECTOR)

    # Tracker setup
    TRACKER = tr.setup_tracker(config.PANCAKE)

    # Input data setup
    source = config.PANCAKE.DATA.SOURCE
    source_path = fix_path(source)

    DATA, is_webcam = load_data(source_path)

    iteration = 0
    for path, im0s, vid_cap in DATA:
        l.debug(f"Iteration {iteration}")
        iteration += 1

        detections = BACKEND.detect(im0s)

        if not type(im0s) is list:
            frame = im0s
        else:
            # TODO: adapt concat to arbitrary number of frames 
            frame = cv2.hconcat([im0s[0], im0s[1], im0s[2]])
        tracks = TRACKER.update(detections, frame)

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
                im0=frame,
                labels=DETECTOR.model.names,
                hide_labels=hide_labels,
                hide_conf=hide_conf,
                line_thickness=line_thickness,
                debug=False,  # Set to True to enable manual stepping
            )


if __name__ == "__main__":
    main()
