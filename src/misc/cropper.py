import math

import cv2
import imutils
import numpy as np

from ..detector.detector_yolo import YOLODetector


# Constants


def f_r(x):
    """Represents centre strip (right side) of Straße des 17. Juni."""
    return int((-1682 / 2335 * x) + (8211072 / 2335))


def f_l(x):
    """Represents centre strip (left side) of Straße des 17. Juni."""
    return int((310 / 441 * x) + (171316 / 441))


L_CONST = {
    "ANGLE": 38,
    "SIDE": 450 // 2,
    "START_X": 1925,
    "END_X": 4000,
    "STEPS": 6,
    "F": f_l,
}

R_CONST = {
    "ANGLE": 360 - 38,
    "SIDE": 416 // 2,
    "START_X": 2900,
    "END_X": 600,
    "STEPS": 7,
    "F": f_r,
}


def partial(
    source,
    side: str = "",
    imshow: bool = False,
    imwrite: bool = False,
    imwrite_filename: str = "results.jpg",
) -> list:
    """Detect objects on image by splitting and merging.

    :param source: filename or np.ndarray
    :param side: right or left side

    :param imshow: show image of results

    :param imwrite: write image of results to disk
    :param imwrite_filename: filename of image of results

    :return objs: list of tuples with objects and their coordinates

    Modus operandi:
    * Rotate image (without cropping)
    * Select subframes
    * Detect objects on each subframe, saving classes and coordinates
    * Calculate coordinates on original frame
    """
    if type(source) is str:
        img = cv2.imread(source)
    elif type(source) is not np.ndarray:
        raise ValueError("Source needs to be str or np.ndarray")

    valid_side = ["l", "left", "r", "right"]
    if not side or side not in valid_side:
        raise ValueError(f"param side needs to be one of {valid_side}")
    if side.startswith("l"):
        CONST = L_CONST
    else:
        CONST = R_CONST

    img = imutils.rotate_bound(img, CONST["ANGLE"])

    subframes = []
    for x in range(
        CONST["START_X"],
        CONST["END_X"],
        -((CONST["START_X"] - CONST["END_X"]) // CONST["STEPS"]),
    ):
        y = CONST["F"](x)
        CONST["SIDE"] = int(1.1 * CONST["SIDE"])
        subframe = img[
            y - CONST["SIDE"] : y + CONST["SIDE"], x - CONST["SIDE"] : x + CONST["SIDE"]
        ]
        subframes.append(subframe)
        cv2.rectangle(
            img,
            (x - CONST["SIDE"], y - CONST["SIDE"]),
            (x + CONST["SIDE"], y + CONST["SIDE"]),
            (0, 0, 255),
            5,
        )

    detector = YOLODetector()
    for subframe in subframes:
        res = detector.detect(subframe)
        print(res)
        # TODO: stuff

    if imshow:
        cv2.namedWindow("results", cv2.WINDOW_NORMAL)
        cv2.imshow("results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if imwrite:
        cv2.imwrite(imwrite_filename, img)

    return subframes


def right():
    raise DeprecationWarning
    global img
    filename = "../../samples/images/random2_4k/1r.jpg"
    img = cv2.imread(filename)
    img = imutils.rotate_bound(img, 360 - 38)
    # Crops

    side = 416 // 2
    start_x = 2900
    end_x = 600
    steps = 7
    for x in range(start_x, end_x, -((start_x - end_x) // steps)):
        y = f(x)
        side = int(1.1 * side)
        cv2.rectangle(img, (x - side, y - side), (x + side, y + side), (0, 0, 255), 5)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.imwrite("FOOO.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def left():
    raise DeprecationWarning
    global img
    filename = "../../samples/images/random2_4k/1l.jpg"
    img = cv2.imread(filename)
    img = imutils.rotate_bound(img, 38)
    # Crops

    side = 450 // 2
    start_x = 1925
    end_x = 4000
    steps = 6
    for x in range(start_x, end_x, -((start_x - end_x) // steps)):
        y = f(x)
        side = int(1.1 * side)
        cv2.rectangle(img, (x - side, y - side), (x + side, y + side), (0, 0, 255), 5)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.imwrite("FOOO.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    partial("../../samples/images/random2_4k/1l.jpg", side="l")
