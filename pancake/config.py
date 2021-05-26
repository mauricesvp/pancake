from .utils.parser import get_config
import configparser
import os


def pancake_config() -> configparser.ConfigParser:
    """Parse config file."""
    # Using pancake.cfg
    # config = configparser.ConfigParser()
    # basedir = os.path.dirname(__file__)
    # config_path = os.path.join(basedir, "pancake.cfg")
    # config.read(config_path)
    # return config

    # Using pancake.yaml
    basedir = os.path.dirname(__file__)
    config_path = os.path.join(basedir, "pancake.yaml")
    return get_config(config_path)
