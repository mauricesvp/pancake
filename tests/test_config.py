from pancake import config


def test_config():
    conf = config.get_config()
    assert conf.items("GENERAL")
    assert conf.items("DATA")
    assert conf.items("DETECTOR")
    assert conf.items("TRACKER")
    assert conf.items("VISUALIZATION")
    assert conf.items("LOGGING")


if __name__ == "__main__":
    test_config()
