import torch
from .tracker import BaseTracker
from .tracker_deepsort import DEEPSORT
<<<<<<< HEAD
from .tracker_centroid import CentroidTracker
=======
from ..utils.common import fix_path
from ..utils.parser import get_config

__all__ = ["TRACKER_REGISTRY", "setup_tracker"]

>>>>>>> 0969e9564df0b228c4e5d70c92c09bbd2d9c63ac

TRACKER_REGISTRY = BaseTracker.get_subclasses()


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

    TRACKER = TRACKER_REGISTRY[name](tracker_cfg, device=device)
    return TRACKER
