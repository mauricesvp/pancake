import configparser


def test_configparser():
    config = configparser.ConfigParser()
    config.read("test.cfg")
    print(config.items("SPECIFIC"))


if __name__ == "__main__":
    test_configparser()
