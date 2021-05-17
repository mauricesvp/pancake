import glob
import os

from pancake.misc.detect_wrapper import DetectWrapper
from pancake.detector.detector_yolo import YOLODetector


# Helper functions

def same_shape(*imgs: list) -> bool:
    init = imgs[0].shape
    for other in imgs[1:]:
        if other.shape != init:
            return False
    return True


def test_series():
    """Test using series of images."""
    # Get images
    imgs = glob.glob("../samples/images/random3_4k/*/*jpg")
    each = len(imgs)//3
    imgslist = [(imgs[i], imgs[i+each], imgs[i+2*each]) for i in range(each)]

    # Init detector and wrapper
    det = YOLODetector()
    dw = DetectWrapper(det)

    # Run
    results = []
    for c, l, r in imgslist:  # c, l, r because of alphabetic order
        dw.run_detection(l, c, r)
        assert dw.result
        results += dw.result
    with open("results.txt", "w+") as f:
        f.write(str(results))

    # TODO: Add more asserts here


def test_basic():
    """Basic test using only one timeframe."""
    l = "../samples/images/random4_4k/1l.jpg"
    c = "../samples/images/random4_4k/1c.jpg"
    r = "../samples/images/random4_4k/1r.jpg"

    det = YOLODetector()
    dw = DetectWrapper(det, write_partials=True)
    dw.run_detection(l, c, r)
    dw.write("result.jpg")
    # TODO: Add more asserts here
    assert dw.result


if __name__ == "__main__":
    test_series()
