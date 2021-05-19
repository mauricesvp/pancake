from pancake import logger


def test_logger():
    """
    The proper way of testing loggers would be this:
    https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertLogs
    """
    l = logger.setup_logger("test")
    assert l
    assert l.debug
    assert l.info
    assert l.warning
    assert l.error
    assert l.critical


if __name__ == "__main__":
    test_logger()
