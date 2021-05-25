import configparser
import os


def get_config() -> configparser.ConfigParser:
    """Parse config file."""
    config = configparser.ConfigParser()
    basedir = os.path.dirname(__file__)
    config_path = os.path.join(basedir, "pancake.cfg")
    config.read(config_path)
    return config
