"""
Divide Et Impera (Divide And Conquer) Backend
----------------
TODOs:
    * Apply ROIs
    * Filter by object class id
      (this could be done in the detector as well)

"""
import math
import time
from functools import lru_cache

import cv2
import imutils
import numpy as np
import torch

from shapely.geometry import Polygon

from .backend import Backend
from pancake.logger import setup_logger

l = setup_logger(__name__)


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
    locations = []
    if not (x0 > 0 and y0 > 0):
        return locations
    for i, subframe in enumerate(subframes):
        tlx, tly, brx, bry = subframe[:4]
        # Check if any corner is in the subframe
        if x0 >= tlx and x0 <= brx and y0 >= tly and y0 <= bry:  # tl
            locations.append(i)
            continue
        if x1 >= tlx and x1 <= brx and y0 >= tly and y0 <= bry:  # tr
            locations.append(i)
            continue
        if x1 >= tlx and x1 <= brx and y1 >= tly and y1 <= bry:  # br
            locations.append(i)
            continue
        if y1 >= tly and y1 <= bry and x0 >= tlx and x0 <= brx:  # bl
            locations.append(i)
            continue
    return locations


def hconcat(source):
    """Concat images horizontally"""
    if type(source) != list:
        source = [source]
    return cv2.hconcat(source)


def rev_rotate_bound(img: np.ndarray, result: tuple, angle: int, side: int) -> list:
    h, w, _ = img.shape
    center = (w // 2, h // 2)

    x0, y0, x1, y1, cf, cl = result
    # Reverse rotation using rotation matrix
    # Note that we will from this point on use all four corners!
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    ntl = M.dot([x0, y0, 1])
    nbr = M.dot([x1, y1, 1])
    ntr = M.dot([x1, y0, 1])
    nbl = M.dot([x0, y1, 1])

    points = [*ntl, *nbr, *ntr, *nbl]
    xmax = max(list(points[:8:2]))
    xmin = min(list(points[:8:2]))
    ymax = max(list(points[1:8:2]))
    ymin = min(list(points[1:8:2]))

    diff = (w - side) // 2

    tlx, tly = int(xmin - diff), int(ymin - diff)
    brx, bry = int(xmax - diff), int(ymax - diff)

    return (tlx, tly, brx, bry, cf, cl)


def rotate_cpu(image, angle):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def rotate(image, angle):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_cuda = cv2.cuda_GpuMat()
    img_cuda.upload(image)

    rotated = cv2.cuda.warpAffine(img_cuda, M, (w, h))
    return rotated.download()


def rotate_bound_cpu(img: np.ndarray, angle: int):
    """
    Source:
        github.com/jrosebr1/imutils/blob/master/imutils/convenience.py#L41-L63.
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (nW, nH))


def rotate_bound(img: np.ndarray, angle: int):
    """
    Source:
        github.com/jrosebr1/imutils/blob/master/imutils/convenience.py#L41-L63.
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    img_cuda = cv2.cuda_GpuMat()
    img_cuda.upload(img)

    res = cv2.cuda.warpAffine(img_cuda, M, (nW, nH))
    return res.download()


# Constants


@lru_cache(maxsize=None)
def f(x: int) -> (int, int):
    """Returns y value, angle of rotation, width for x along centre strip."""
    if x < 3840:
        y = int(-0.047 * x + 1100)
        angle = int(-0.015 * x + 57)
        w = int(0.216 * x + 172)
    elif x < 7627:
        y = 920
        angle = 0
        w = 1000
    else:
        y = int(0.054 * x + 510.8)
        angle = 360 - int(0.013 * x - 100.3)
        w = int(-0.213 * x + 2626)
    return y, angle, w // 2


CONST = {
    "START_X": 1280,
    "END_X": 10515,
    "STEPS": 21,  # Deprecated
    "F": f,
}


class DEI(Backend):
    def __init__(
        self,
        detector,
        roi: list = None,
        simple: bool = False,
        cache: bool = True,
        config: dict = {},
        *args,
        **kwargs,
    ) -> None:
        """

        :param detector: Detector which provides 'detect' method,
                         which can take one or multiple images.
        :param simple (bool): Use less subframes

        """
        self.detector = detector
        self.simple = simple

        if roi:
            self.roi = roi

        # Check if we can use cv2.cuda
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.rotate = rotate
            self.rotate_bound = rotate_bound
        else:
            self.rotate = rotate_cpu
            self.rotate_bound = rotate_bound_cpu
        # Note that rotate(_cpu) is not used as of now

        if config:
            self.simple = config["SIMPLE"]

        self.cache = cache
        if cache:
            xyas = []
            x = CONST["START_X"]
            y, angle, side = f(x)
            if self.simple:
                side = int(1.7 * side)
            xyas.append([x, y, angle, side])
            while x < (CONST["END_X"] + side):
                x = x + int(1.5 * side)
                y, angle, side = f(x)
                if self.simple:
                    side = int(1.7 * side)
                xyas.append([x, y, angle, side])
            self.xyas = xyas

    def rotate(self):
        pass

    def rotate_bound(self):
        pass

    def detect(
        self,
        source,
    ) -> (list, np.ndarray):
        """
        Detect objects on image by splitting and merging.

        :param source: filename or np.ndarray

        :return (objs, img): list of tuples with objects and their coordinates,
                             stitched panorama image

        The tuples of the return list have 6 values:
        x0: x-value top left corner
        y0: y-value top left corner
        x1: x-value bottom right corner
        y1: y-value bottom right corner
        conf: Confidence of detection, values between 0 and 1
        class id: Integer indicating the detected object type

        Modus operandi:
        * Stitch images
        * Divide image into subframes
        * Rotate subframes (without cropping, i.e. adding padding)
        * Detect objects on each subframe, saving classes and coordinates
        * Calculate coordinates on original frame
        * Filter and merge objects
        """
        assert all(x.shape == source[0].shape for x in source[1:])

        # Crop center (fix overlapping)
        h, w, _ = source[1].shape
        crop = round(0.007 * w)  # Width to remove left and right
        source[1] = source[1][0:h, crop : w - crop]

        img = hconcat(source)

        if self.cache:
            subframes = []
            for x, y, angle, side in self.xyas:
                tlx, tly, brx, bry = x - side, y - side, x + side, y + side
                # TODO: ROI
                # tly = max(500, tly)
                # bry = min(1350, bry)
                subframe = img[tly:bry, tlx:brx]
                rot = self.rotate_bound(subframe, angle)
                subframes.append((rot, tlx, tly, brx, bry, angle, side))
        else:
            x = CONST["START_X"]
            y, angle, side = f(x)
            subframes = []
            while x < (CONST["END_X"] + side):
                if self.simple:
                    side = int(1.7 * side)
                tlx, tly, brx, bry = x - side, y - side, x + side, y + side
                subframe = img[tly:bry, tlx:brx]
                rot = self.rotate_bound(subframe, angle)
                subframes.append((rot, tlx, tly, brx, bry, angle, side))
                x = x + int(1.5 * side)
                y, angle, side = f(x)

        imgs = [x[0] for x in subframes]

        result = self.detector.detect(imgs)

        objs = []
        for i, sub in enumerate(result):
            tlx, tly = subframes[i][1:3]
            for obj in sub:
                if subframes[i][5] != 0:  # angle
                    obj = rev_rotate_bound(
                        subframes[i][0], obj, subframes[i][5], subframes[i][6] * 2
                    )
                x0, y0, x1, y1, conf, classid = obj
                x0, y0, x1, y1, conf, classid = (
                    int(x0),
                    int(y0),
                    int(x1),
                    int(y1),
                    float(conf),
                    int(classid),
                )
                # top left
                rtlx, rtly = tlx + x0, tly + y0
                # bottom right
                rbrx, rbry = tlx + x1, tly + y1
                # save coords, conf, class
                objs.append((rtlx, rtly, rbrx, rbry, conf, classid, i))

        subcoords = [x[1:5] for x in subframes]
        results = self.merge(objs, subcoords)

        if False:  # Debugging
            for obj in results:
                x0, y0, x1, y1, *_ = obj
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
            for subf in subframes:
                x0, y0, x1, y1 = subf[1:5]
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.imwrite("stuff.jpg", img)

        for i, x in enumerate(results):
            results[i] = torch.FloatTensor(list(x[:6]))

        results = torch.stack(results, dim=0) if results else torch.empty((0, 6))
        return results, img

    def merge(self, objs: list, subframes: list = None, ratio=0.8) -> list:
        """
        Merge all detected objects of subframes.

        Right now we use a single heuristic:
        If obj is embedded in another object with 90% or more of its area, discard it.

        TODO:
            This simple heuristic is by no means perfect yet.
            Especially when an object is right at the border of a subframe,
            the results can get inaccurate.
            Further filtering has to be done here
            Example:
            * If obj is at the border of a subframe, check if it is in the middle
              of another one (and discard if so)
        """
        results = []
        while objs:
            obj = objs.pop(0)
            # First, check if obj is present in multiple subframes
            # If not, we can add it directly
            if subframes:
                locations = locate(subframes, *obj[:4])
                if len(locations) == 0:  # This should _never_ happen
                    l.warn("Object not located in any subframe!")
                    continue
                if len(locations) == 1:
                    results.append(obj)
                    continue
            # If yes, we need to do some filtering

            def intersect_area(a: tuple, b: tuple):
                dx = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])
                dy = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])
                if dx >= 0 and dy >= 0:
                    return np.maximum(0, dx * dy)
                return 0

            def embedded(obj: tuple, objs: list, results: list):
                # Check if object is embedded in other object (with >80% of its area)
                x0, y0, x1, y1 = obj[:4]
                subframe = obj[6]
                rect1 = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                objarea = (x1 - x0) * (y1 - y0)
                skip = False
                for objtemp in objs + results:
                    if np.abs(objtemp[6] - subframe) > 1:
                        continue
                    ia = intersect_area(obj[:4], objtemp[:4])
                    if ia >= (ratio * objarea):
                        # Our current obj is embedded, skip
                        return True
                return False

            skip = embedded(obj, objs, results)

            if not skip:
                results.append(obj)

        return results
