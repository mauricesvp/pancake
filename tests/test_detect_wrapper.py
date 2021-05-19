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


def test_series(write: bool = False):
    """Test using series of images."""
    # Get images
    this = os.path.dirname(__file__)
    path = os.path.join(this, "../samples/images/random3_4k/*/*jpg")
    imgs = glob.glob(path)
    each = len(imgs) // 3
    imgslist = [(imgs[i], imgs[i + each], imgs[i + 2 * each]) for i in range(each)]
    imgslist = imgslist[:1]

    # Init detector and wrapper
    det = YOLODetector()
    dw = DetectWrapper(det, write_partials=False)

    # Run
    results = []
    for c, l, r in imgslist:  # c, l, r because of alphabetic order
        timestamp = os.path.basename(c).replace(".jpg", "")
        det_res = dw.run_detection(l, c, r, imwrite_interim=False)
        assert dw.result  # Double check
        results.append((*det_res, timestamp))

    if write:
        with open("sample_results.py", "w+") as f:
            f.write("sample_results=")
            f.write(str(results))

    # TODO: Add more asserts here


def test_basic():
    """Basic test using one timestamp."""
    this = os.path.dirname(__file__)
    l = "../samples/images/random4_4k/1l.jpg"
    c = "../samples/images/random4_4k/1c.jpg"
    r = "../samples/images/random4_4k/1r.jpg"
    pathl = os.path.join(this, l)
    pathc = os.path.join(this, c)
    pathr = os.path.join(this, r)

    det = YOLODetector()
    dw = DetectWrapper(det, write_partials=False)
    dw.run_detection(pathl, pathc, pathr)
    # dw.write("result.jpg")
    # TODO: Add more asserts here
    assert dw.result


if __name__ == "__main__":
    test_series()
    # test_basic()
