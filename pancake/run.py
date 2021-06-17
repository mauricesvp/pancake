import argparse
import logging
import time

from . import detector as det
from . import tracker as tr
from .detector import backends as be

from .config import pancake_config
from .logger import setup_logger
from .utils.common import fix_path, load_data, setup_result_processor

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

    # Result processor setup
    RESULT_PROC = setup_result_processor(
        config.PANCAKE.RESULT_PROCESSOR, DETECTOR.model.names
    )

    t1, t2 = 0, 0
    iteration = 0
    for path, im0s, vid_cap in DATA:
        l.debug(f"Iteration {iteration}")
        iteration += 1

        detections, frame = BACKEND.detect(im0s)

        tracks = TRACKER.update(detections, frame)

        RESULT_PROC.process(detections, tracks, frame)

        if n and iteration >= n:
            break

        t1 = time.time()
        l.info(f"--> approx. RUN FPS: {int(1/(t1-t2))}")
        t2 = t1
        

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", nargs="?", type=str, default=None, help="pancake config path"
    )
    args = args = parser.parse_args()

    main(args.cfg)
