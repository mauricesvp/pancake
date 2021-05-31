from .utils.parser import get_config
import configparser
import os


def pancake_config(cfg_path: str = None) -> configparser.ConfigParser:
    """Parse config file."""
    # Using pancake.cfg
    # config = configparser.ConfigParser()
    # basedir = os.path.dirname(__file__)
    # config_path = os.path.join(basedir, "pancake.cfg")
    # config.read(config_path)
    # return config

    # Using pancake.yaml
    if cfg_path is None:
        basedir = os.path.dirname(__file__)
        cfg_path = os.path.join(basedir, "pancake.yaml")
    return get_config(cfg_path)
