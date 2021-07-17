from typing import Type

import torch
from .tracker import BaseTracker
from .tracker_deepsort import DEEPSORT
from .tracker_centroid import CentroidTracker
from ..utils.common import fix_path
from ..utils.parser import get_config

__all__ = ["TRACKER_REGISTRY", "setup_tracker"]


TRACKER_REGISTRY = BaseTracker.get_subclasses()


def setup_tracker(config: dict) -> Type[BaseTracker]:
    """ Helper function to set up a tracker specified in the configurations.

    Args:
        config (dict): Dictionary containing configurations.

    Returns:
        Type[BaseTracker]: A Tracker subclass instance.
    """    
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
