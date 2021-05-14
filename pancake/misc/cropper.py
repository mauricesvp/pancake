import math

import cv2
import imutils
import numpy as np

from shapely.geometry import Polygon

from ..detector.detector_yolo import YOLODetector


# Helper functions


def locate(subframes, x0, y0, x1, y1) -> list:
    """Return all subframe ids the obj is present in.

    This function assumes the coordinates are laid out like this:
    x0, y0 ------------
       |               |
       |               |
       |               |
        ------------ x1, y1
    """
    if not (x0 > 0 and y0 > 0):
        return []
    locations = []
    for i, subframe in enumerate(subframes):
        tlx, tly, brx, bry = subframe[:4]
        if x0 >= tlx and y0 >= tly and x1 <= brx and y1 <= bry:
            locations.append(i)
    return locations


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

    The tuples of the return list have 6 values:
    x0: x-value top left corner
    y0: y-value top left corner
    x1: x-value bottom right corner
    y1: y-value bottom right corner
    conf: Confidence of detection, values between 0 and 1
    class id: Integer indicating the detected object type

    Modus operandi:
    * Rotate image (without cropping)
    * Divide image into subframes
    * Detect objects on each subframe, saving classes and coordinates
    * Dedup objects
    * Calculate coordinates on original frame
    """
    if type(source) is str:
        img = cv2.imread(source)
        if img is None:
            raise ValueError("File does not seem to exist")
    elif type(source) is not np.ndarray:
        raise ValueError("Source needs to be str or np.ndarray")

    valid_side = ["l", "left", "r", "right"]
    if not side or side not in valid_side:
        raise ValueError(f"param side needs to be one of {valid_side}")
    if side.startswith("l"):
        CONST = L_CONST
    else:
        CONST = R_CONST

    detector = YOLODetector()

    img = imutils.rotate_bound(img, CONST["ANGLE"])

    # Divide image into subframes
    objs = []
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
        # top left
        tlx, tly = x - CONST["SIDE"], y - CONST["SIDE"]
        # bottom right
        brx, bry = x + CONST["SIDE"], y + CONST["SIDE"]
        subframes.append((tlx, tly, brx, bry))
        cv2.rectangle(
            img,
            (tlx, tly),
            (brx, bry),
            (0, 0, 255),
            5,
        )

        res = detector.detect(subframe)[0]
        for obj in res:
            x0, y0 = obj[:2]
            x1, y1 = obj[2:4]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            # real points
            # top left
            rtlx, rtly = tlx + x0, tly + y0
            # bottom right
            rbrx, rbry = tlx + x1, tly + y1
            # save coords, conf, class
            objs.append((rtlx, rtly, rbrx, rbry, obj[4], obj[5]))

    results = []
    while objs:
        obj = objs.pop(0)
        # First, check if obj is present in multiple subframes
        # If not, we can add it directly
        locations = locate(subframes, *obj[:4])
        if len(locations) == 0:
            continue  # This should _never_ happen
        if len(locations) == 1:
            results.append(obj)
            continue
        # If yes, we need to do some filtering
        # We can go about this more or less sophisticated.
        # Naturally, we'll go with the easy way first.

        # Check if object is embedded in other object (with >80% of its area)
        emb = 0.80
        x0, y0, x1, y1 = obj[:4]
        rect1 = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        skip = False
        for objtemp in objs + results:
            xt0, yt0, xt1, yt1 = objtemp[:4]
            rect2 = Polygon([(xt0, yt0), (xt1, yt0), (xt1, yt1), (xt0, yt1)])
            intersection = rect1.intersection(rect2)
            if intersection.area >= (emb * rect1.area):
                # Our current obj is embedded, skip
                skip = True
                break

        if not skip:
            results.append(obj)

        # TODO: This simple heuristic is by no means perfect yet
        # Especially when an object is right at the border of a subframe,
        # the results can get inaccurate
        # Further filtering has to be done here

    # TODO: Reverse rotation

    # Draw results on image
    if imshow or imwrite:
        for obj in results:
            tlx, tly, brx, bry = obj[:4]
            cv2.rectangle(
                img,
                (tlx, tly),
                (brx, bry),
                (255, 0, 0),
                1,
            )

    if imshow:
        cv2.namedWindow("results", cv2.WINDOW_NORMAL)
        cv2.imshow("results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if imwrite:
        cv2.imwrite(imwrite_filename, img)

    return results


if __name__ == "__main__":
    partial("../../samples/images/random2_4k/1l.jpg", side="l")
