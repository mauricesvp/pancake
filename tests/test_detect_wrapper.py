from pancake.misc.detect_wrapper import DetectWrapper
from pancake.detector.detector_yolo import YOLODetector


def test_detect_wrapper():
    l = "../samples/images/random2_4k/1l.jpg"
    c = "../samples/images/random2_4k/1c.jpg"
    r = "../samples/images/random2_4k/1r.jpg"

    l = "../samples/images/random4_4k/1l.jpg"
    c = "../samples/images/random4_4k/1c.jpg"
    r = "../samples/images/random4_4k/1r.jpg"

    det = YOLODetector()
    dw = DetectWrapper(det, write_partials=True)
    dw.run_detection(l, c, r)
    dw.write("result.jpg")
    # TODO: Add asserts here
    assert dw.result


if __name__ == "__main__":
    test_detect_wrapper()
