import logging

from . import detector as det
from . import tracker as tr

from .config import pancake_config
from .logger import setup_logger
from .utils.common import fix_path
from .utils.parser import get_config

l = setup_logger(__name__)


def setup_detector(config):
    name = config["DETECTOR"]["detector"]
    params = config["DETECTOR_" + name.upper()]

    device = config["GENERAL"]["device"]

    DETECTOR = det.DETECTOR_REGISTRY[name](params, device=device)
    return DETECTOR


def setup_tracker(config):
    name = config["TRACKER"]["tracker"]
    conf = config["TRACKER_" + name.upper()]
    tracker_cfg = get_config(config_file=conf["tracker_cfg_path"])

    device = config["GENERAL"]["device"]

    TRACKER = tr.TRACKER_REGISTRY[name](tracker_cfg, device=device)
    return TRACKER


def setup_logging(config):
    try:
        log_level = getattr(logging, config["level"])
        l.setLevel(log_level)
    except:
        l.setLevel(logging.INFO)
    l.debug(f"Log level set to {log_level}.")


def main():
    l.debug("Starting pancake.")

    config = pancake_config()

    setup_logging(config["LOGGING"])

    # Detector setup
    DETECTOR = setup_detector(config)

    # Tracker setup
    TRACKER = setup_tracker(config)
    # tracker_cfg = fix_path(tracker_cfg_path)
    # TRACKER = tr.TRACKER_REGISTRY[tracker](tracker_cfg, device=device)

    # Input data setup
    source = config["DATA"]["source"]
    source_path = fix_path(source)

    # !!! TODO: Make load_data _not_ require model param !!!
    # DATA, is_webcam = load_data(source, MODEL)


if __name__ == "__main__":
    main()
