"""
Divide Et Impera (Divide And Conquer) Backend
----------------
TODOs:
    * Apply ROIs
    * Filter by object class id
      (this could be done in the detector as well)

"""
import math

import cv2
import imutils
import numpy as np
import torch

from shapely.geometry import Polygon

from .backend import Backend


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
    """Convert detection results to integer values.
    Note that confidence is a float value however."""
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
    "SIDE": 448 // 2,
    "START_X": 1925,
    "END_X": 4000,
    "STEPS": 6,
    "F": f_l,
}

R_CONST = {
    "ANGLE": 360 - 38,
    "SIDE": 384 // 2,
    "START_X": 2900,
    "END_X": 600,
    "STEPS": 7,
    "F": f_r,
}


class DEI(Backend):
    result: list  # List of tuples, intended for tracker
    imgs: list  # List of image parts
    img: np.ndarray  # Stitched panorama image
    printed: np.ndarray  # Stitched image with detection results

    def __init__(self, detector, write_partials: bool = False, *args, **kwargs) -> None:
        """

        :param detector: Detector which provides 'detect' method,
                         which can take one or multiple images.

        """
        self.detector = detector
        self.write_partials = write_partials

    def detect(
        self,
        source,
        imwrite_interim: bool = False,
        imwrite_interim_filename: str = "partial_interim.jpg",
    ) -> list:
        """Detect objects on Panorama images."""
        assert len(source) == 3
        imgl = self.get_img(source[0])
        imgc = self.get_img(source[1])
        imgr = self.get_img(source[2])
        assert imgl.shape == imgc.shape == imgr.shape
        self.shape = imgl.shape
        objs = []
        # objs += self.partial(
        objs += self.partial_non_batch(
            imgl,
            side="l",
            imwrite=self.write_partials,
            imwrite_interim=imwrite_interim,
            imwrite_interim_filename=imwrite_interim_filename,
        )
        # rights = self.partial(
        rights = self.partial_non_batch(
            imgr,
            side="r",
            imwrite=self.write_partials,
            imwrite_interim=imwrite_interim,
            imwrite_interim_filename=imwrite_interim_filename,
        )
        mids = self.detect_mid(imgc)
        # Add offsets to mid/right
        h, w, _ = imgc.shape
        mids = [
            (a + w, b, c + w, b, c + w, d, a + w, d, e, f) for a, b, c, d, e, f in mids
        ]
        rights = [
            (a + (2 * w), b, c + (2 * w), d, e + (2 * w), f, g + (2 * w), h, i, j)
            for a, b, c, d, e, f, g, h, i, j in rights
        ]
        objs += mids + rights

        # Un-rotate left/right
        for i, x in enumerate(objs):
            if len(x) == 10:
                xs = list(x[:8:2])
                ys = list(x[1:8:2])
                tlx = min(xs)
                tly = min(ys)
                brx = max(xs)
                bry = max(ys)
                objs[i] = (tlx, tly, brx, bry, x[8], x[9])

        # TODO: Apply ROIs here

        for i, x in enumerate(objs):
            objs[i] = torch.FloatTensor(list(x))
        objs = torch.stack(objs, dim=0)

        self.result = objs
        self.imgs = [imgl, imgc, imgr]
        self.set_full_img()  # set self.img
        return objs

    def detect_mid(self, imgc) -> list:
        # imgc = cv2.resize(imgc, (448*4, 448*4))
        # imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        res = self.detector.detect(imgc)[0]
        return res2int(res)

    def partial_non_batch(
        self,
        source,
        side: str = "",
        imshow: bool = False,
        imwrite: bool = False,
        imwrite_filename: str = "partial.jpg",
        imwrite_interim: bool = False,
        imwrite_interim_filename: str = "partial_interim.jpg",
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
        * Rotate image (without cropping, i.e. adding padding)
        * Divide image into subframes
        * Detect objects on each subframe, saving classes and coordinates
        * Filter and merge objects
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

        orig_img = img.copy()
        img = imutils.rotate_bound(img, CONST["ANGLE"])

        # Divide image into subframes
        res = []
        objs = []
        subframes = []
        subframes_imgs = []
        const_tmp = CONST["SIDE"]
        for x in range(
            CONST["START_X"],
            CONST["END_X"],
            -((CONST["START_X"] - CONST["END_X"]) // CONST["STEPS"]),
        ):
            y = CONST["F"](x)
            const_tmp = int(1.1 * const_tmp)
            # top left
            tlx, tly = x - const_tmp, y - const_tmp
            # bottom right
            brx, bry = x + const_tmp, y + const_tmp
            subframe = img[
                tly:bry,
                tlx:brx,
            ]
            subframes.append((tlx, tly, brx, bry))
            subframes_imgs.append((subframe, tlx, tly))
            cv2.rectangle(
                img,
                (tlx, tly),
                (brx, bry),
                (0, 0, 255),
                5,
            )

            res += self.detector.detect(subframe)

        # Get real points
        for i, sub in enumerate(res):
            tlx, tly = subframes_imgs[i][1:]
            for obj in sub:
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
                objs.append((rtlx, rtly, rbrx, rbry, conf, classid))

        if imwrite_interim:
            img_interim = self.draw(img, objs)
            # Also draw center point for good measure
            for obj in subframes:
                center = (np.mean([obj[0], obj[2]]), np.mean([obj[1], obj[3]]))
                center = (int(center[0]), int(center[1]))
                cv2.circle(img_interim, center, 2, (255, 0, 0))
            cv2.imwrite(side + imwrite_interim_filename, img_interim)

        # Merge objects on subframes
        results = self.merge(objs, subframes)

        # Reverse rotation
        # All object tuples now have all 4 corners!
        results = self.rev_rotate(img, results, CONST["ANGLE"])

        # Draw results on image
        if imshow or imwrite:
            orig_img = self.draw(orig_img, results)

        if imshow:
            cv2.namedWindow("results", cv2.WINDOW_NORMAL)
            cv2.imshow("results", orig_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if imwrite:
            if imwrite_filename == "partial.jpg":
                imwrite_filename = f"partial_{side}.jpg"
            cv2.imwrite(imwrite_filename, orig_img)

        return results

    def partial(
        self,
        source,
        side: str = "",
        imshow: bool = False,
        imwrite: bool = False,
        imwrite_filename: str = "partial.jpg",
        imwrite_interim: bool = False,
        imwrite_interim_filename: str = "partial_interim.jpg",
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
        * Rotate image (without cropping, i.e. adding padding)
        * Divide image into subframes
        * Detect objects on each subframe, saving classes and coordinates
        * Filter and merge objects
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

        orig_img = img.copy()
        img = imutils.rotate_bound(img, CONST["ANGLE"])

        # Divide image into subframes
        objs = []
        subframes = []
        subframes_imgs = []
        const_tmp = CONST["SIDE"]
        for x in range(
            CONST["START_X"],
            CONST["END_X"],
            -((CONST["START_X"] - CONST["END_X"]) // CONST["STEPS"]),
        ):
            y = CONST["F"](x)
            const_tmp = int(1.1 * const_tmp)
            # top left
            tlx, tly = x - const_tmp, y - const_tmp
            # bottom right
            brx, bry = x + const_tmp, y + const_tmp
            subframe = img[
                tly:bry,
                tlx:brx,
            ]
            subframes.append((tlx, tly, brx, bry))
            subframes_imgs.append((subframe, tlx, tly))
            cv2.rectangle(
                img,
                (tlx, tly),
                (brx, bry),
                (0, 0, 255),
                5,
            )

        # Run batch detection on subframes
        imgs = [x[0] for x in subframes_imgs]
        # imgs = np.array([np.array(x[0]).astype("int64") for x in subframes_imgs])
        # imgs = np.stack(imgs, axis=0)
        # for i, x in enumerate(subframes_imgs):
        # subframes_imgs[i] = torch.FloatTensor(list(x[0]))
        res = self.detector.detect(imgs)

        # Get real points
        for i, sub in enumerate(res):
            tlx, tly = subframes_imgs[i][1:]
            for obj in sub:
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
                objs.append((rtlx, rtly, rbrx, rbry, conf, classid))

        if imwrite_interim:
            img_interim = self.draw(img, objs)
            # Also draw center point for good measure
            for obj in subframes:
                center = (np.mean([obj[0], obj[2]]), np.mean([obj[1], obj[3]]))
                center = (int(center[0]), int(center[1]))
                cv2.circle(img_interim, center, 2, (255, 0, 0))
            cv2.imwrite(side + imwrite_interim_filename, img_interim)

        # Merge objects on subframes
        results = self.merge(objs, subframes)

        # Reverse rotation
        # All object tuples now have all 4 corners!
        results = self.rev_rotate(img, results, CONST["ANGLE"])

        # Draw results on image
        if imshow or imwrite:
            orig_img = self.draw(orig_img, results)

        if imshow:
            cv2.namedWindow("results", cv2.WINDOW_NORMAL)
            cv2.imshow("results", orig_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if imwrite:
            if imwrite_filename == "partial.jpg":
                imwrite_filename = f"partial_{side}.jpg"
            cv2.imwrite(imwrite_filename, orig_img)

        return results

    def rev_rotate(self, img: np.ndarray, results: list, angle: int) -> list:
        h, w, _ = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        tmpimg = imutils.rotate(img, angle)
        htmp, wtmp, _ = tmpimg.shape
        # original image(s) shape
        h, w, _ = self.shape
        xdiff = (wtmp - w) // 2
        ydiff = (htmp - h) // 2

        for i, obj in enumerate(results):
            x0, y0, x1, y1, cf, cl = obj
            area = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).area
            ar = (x1 - x0) // (y1 - y0)  # aspect ratio
            # Reverse rotation using rotation matrix
            # Note that we will from this point on use all four corners!
            ntl = M.dot([x0, y0, 1])
            nbr = M.dot([x1, y1, 1])
            ntr = M.dot([x1, y0, 1])
            nbl = M.dot([x0, y1, 1])

            allcords = [ntl] + [ntr] + [nbr] + [nbl]
            allcords = [[int(a), int(b)] for a, b in allcords]
            allrebased = []
            for point in allcords:
                point[0] -= xdiff
                point[1] -= ydiff
                allrebased += point

            results[i] = (*allrebased, cf, cl)
        return results

    def merge(self, objs: list, subframes: list = None, ratio=0.8) -> list:
        """Merge all detected objects of subframes.

        Right now we use a single heuristic:
        If obj is embedded in another object with 90% or more of its area, discard it.
        """
        results = []
        while objs:
            obj = objs.pop(0)
            # First, check if obj is present in multiple subframes
            # If not, we can add it directly
            if subframes:
                locations = locate(subframes, *obj[:4])
                if len(locations) == 0:
                    continue  # This should _never_ happen
                if len(locations) == 1:
                    results.append(obj)
                    continue
            # If yes, we need to do some filtering

            # Check if object is embedded in other object (with >80% of its area)
            x0, y0, x1, y1 = obj[:4]
            rect1 = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            skip = False
            for objtemp in objs + results:
                xt0, yt0, xt1, yt1 = objtemp[:4]
                rect2 = Polygon([(xt0, yt0), (xt1, yt0), (xt1, yt1), (xt0, yt1)])
                intersection = rect1.intersection(rect2)
                if intersection.area >= (ratio * rect1.area):
                    # Our current obj is embedded, skip
                    skip = True
                    break

            if not skip:
                results.append(obj)

            # TODO: This simple heuristic is by no means perfect yet.
            # Especially when an object is right at the border of a subframe,
            # the results can get inaccurate.
            # Further filtering has to be done here
            # Example:
            # * If obj is at the border of a subframe, check if it is in the middle
            #   of another one (and discard if so)
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

    def draw(self, img: np.ndarray, results: list) -> np.ndarray:
        """Draw results on image."""
        for obj in results:
            if len(obj) > 8:
                tlx, tly, trx, try_, brx, bry, blx, bly = obj[:8]
                pts = np.array(
                    [[x, y] for x, y in zip(obj[:8:2], obj[1:8:2])], dtype=np.int32
                )
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(
                    img,
                    [pts],
                    True,
                    (255, 0, 0),
                    1,
                )
            else:
                tlx, tly, brx, bry = obj[:4]
                cv2.rectangle(
                    img,
                    (int(tlx), int(tly)),
                    (int(brx), int(bry)),
                    (255, 0, 0),
                    1,
                )
        return img

    def show(self) -> None:
        """Show concat of images."""
        if not len(self.printed):
            self.printed = self.draw(self.img, self.results)
        cv2.namedWindow("results", cv2.WINDOW_NORMAL)
        cv2.imshow("results", self.printed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write(self, filename="result.jpg") -> None:
        """Write results to disk."""
        if not len(self.printed):
            self.printed = self.draw(self.img, self.result)
        cv2.imwrite(filename, self.printed)

    def set_full_img(self):
        imgs = []
        for img in self.imgs:
            imgs.append(self.get_img(img))
        self.img = cv2.hconcat(imgs)
