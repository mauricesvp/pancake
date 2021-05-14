import math

import cv2
import imutils
import numpy as np

from shapely.geometry import Polygon


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


def res2int(res):
    """Note that confidence is a float value."""
    objs = []
    for obj in res:
        x0, y0, x1, y1, conf, classid = obj
        x0, y0, x1, y1, conf, classid = (
            int(x0),
            int(y0),
            int(x1),
            int(y1),
            float(conf),
            int(classid),
        )
        objs.append((x0, y0, x1, y1, conf, classid))
    return objs


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


class DetectWrapper:
    result: list  # List of tuples, intended for tracker
    imgs: list  # List of image parts
    img: np.ndarray  # Stitched panorama image
    printed: np.ndarray  # Stichted image with detection results

    def __init__(self, detector, *args, **kwargs) -> None:
        self.detector = detector

    def run_detection(self, imgl, imgc, imgr) -> list:
        """Detect objects on Panorama images."""
        imgl = self.get_img(imgl)
        imgc = self.get_img(imgc)
        imgr = self.get_img(imgr)
        assert imgl.shape == imgc.shape == imgr.shape
        objs = []
        # Note that reversing of rotation is not done yet
        objs += self.partial(imgl, side="l")
        rights = self.partial(imgr, side="r")
        mids = self.detect_mid(imgc)
        # Add offsets to mid/right
        h, w, _ = imgc.shape
        mids = [(a + w, b, c + w, d, e, f) for a, b, c, d, e, f in mids]
        rights = [(a + (2 * w), b, c + (2 * w), d, e, f) for a, b, c, d, e, f in rights]
        objs += mids + rights

        # TODO: Apply ROIs here

        self.result = objs
        self.imgs = [imgl, imgc, imgr]
        self.set_full_img()  # set self.img
        self.draw()  # set self.printed
        return objs

    def detect_mid(self, imgc) -> list:
        res = self.detector.detect(imgc)[0]
        return res2int(res)

    def partial(
        self,
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
        img = self.get_img(source)

        valid_side = ["l", "left", "r", "right"]
        if not side or side not in valid_side:
            raise ValueError(f"param side needs to be one of {valid_side}")
        if side.startswith("l"):
            CONST = L_CONST
        else:
            CONST = R_CONST

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
                y - CONST["SIDE"] : y + CONST["SIDE"],
                x - CONST["SIDE"] : x + CONST["SIDE"],
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

            res = self.detector.detect(subframe)[0]
            for obj in res:
                x0, y0, x1, y1, conf, classid = obj
                x0, y0, x1, y1, conf, classid = (
                    int(x0),
                    int(y0),
                    int(x1),
                    int(y1),
                    int(conf),
                    int(classid),
                )
                # real points
                # top left
                rtlx, rtly = tlx + x0, tly + y0
                # bottom right
                rbrx, rbry = tlx + x1, tly + y1
                # save coords, conf, class
                objs.append((rtlx, rtly, rbrx, rbry, conf, classid))

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

            # TODO: This simple heuristic is by no means perfect yet.
            # Especially when an object is right at the border of a subframe,
            # the results can get inaccurate.
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

    def get_img(self, source) -> np.ndarray:
        if type(source) is str:
            img = cv2.imread(source)
            if img is None:
                raise ValueError("File does not seem to exist")
        elif type(source) is not np.ndarray:
            raise ValueError("Source needs to be str or np.ndarray")
        else:
            img = source
        return img

    def draw(self) -> None:
        """Draw results on image."""
        if not len(self.result) or not len(self.img):
            raise ValueError("Missing results or img, run detection first!")
        img = self.img.copy()
        for obj in self.result:
            tlx, tly, brx, bry = obj[:4]
            assert type(img) == np.ndarray
            cv2.rectangle(
                img,
                (tlx, tly),
                (brx, bry),
                (255, 0, 0),
                1,
            )
        self.printed = img

    def show(self) -> None:
        """Show concat of images."""
        if not len(self.img):
            raise ValueError("Missing img, run detection first!")
        cv2.namedWindow("results", cv2.WINDOW_NORMAL)
        cv2.imshow("results", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write(self, filename="result.jpg", printed: bool = True) -> None:
        """Write results to disk."""
        if not len(self.img) or not len(self.printed):
            raise ValueError("Missing img, run detection first!")
        if printed:
            cv2.imwrite(filename, self.printed)
        else:
            cv2.imwrite(filename, self.img)

    def set_full_img(self):
        imgs = []
        for img in self.imgs:
            imgs.append(self.get_img(img))
        self.img = cv2.hconcat(imgs)
