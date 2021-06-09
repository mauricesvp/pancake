import glob
import os

import cv2

from pancake.detector.backends.backend_dei import DEI
from pancake.detector.detector_yolo_custom import YOLOCustomDetector
from pancake.detector.detector_yolo_simple import YOLOSimpleDetector


def test_backend_dei():
    """Basic test using imgaes of one timestamp."""
    this = os.path.dirname(__file__)
    l = "../samples/images/random4_4k/1l.jpg"
    c = "../samples/images/random4_4k/1c.jpg"
    r = "../samples/images/random4_4k/1r.jpg"
    pathl = os.path.join(this, l)
    pathc = os.path.join(this, c)
    pathr = os.path.join(this, r)
    imgs = []
    for img in [pathl, pathc, pathr]:
        imgs.append(cv2.imread(img))

    # det = YOLOCustomDetector()  # Needs config
    det = YOLOSimpleDetector()

    dei = DEI(det, new=True)
    res = dei.detect(imgs)

    # TODO: Add asserts here


if __name__ == "__main__":
    test_backend_dei()
