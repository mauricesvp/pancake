import cv2
import numpy as np

# Constants (or hard coded values)
# YOU MAY HAVE TO CHANGE VALUES HERE IF RESULTS ARE UNEXPECTED

# Calc bt: bt = final_y // BORDER_BTTM_<LEFT/RIGHT>
BORDER_BTTM_LEFT = 2.376
BORDER_BTTM_RIGHT = 2.483

# ROI (Region of Interest)
# TODO: Make pts relative
PTS_LEFT = [
    [656, 288],
    [1279, 208],
    [1279, 432],
    [537, 397],
]
PTS_RIGHT = [
    [1, 215],
    [925, 331],
    [1006, 388],
    [1, 449],
]


def process_side(img, pts1, border=True, write=True, left=True):
    """Process one side.

    img: input image
    pts1: ROI (Region of Interest)
    border: Fill border around image to match shape of img
    write: Write to file
    left: Left side of image
    """
    final_y, final_x, _ = img.shape
    # Max distance of width, height
    # In other words, how large is the smallest rectangle that fits around all 4 points of pts1
    xs = [x[0] for x in pts1]
    ys = [x[1] for x in pts1]
    mw = int(max(xs) - min(xs))
    mh = int(max(ys) - min(ys))
    pts2 = [
        [0, 0],
        [mw, 0],
        [mw, mh],
        [0, mh],
    ]
    pts2 = np.float32(pts2)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (mw, mh))

    if border:
        factor = BORDER_BTTM_LEFT if left else BORDER_BTTM_RIGHT
        bt = round(final_y // factor)
        tp = round(final_y - bt - mh)
        if left:
            lt = final_x - mw
            rt = 0
        else:
            lt = 0
            rt = final_x - mw
        # Fill up missing space
        img = cv2.copyMakeBorder(
            img, tp, bt, lt, rt, cv2.BORDER_CONSTANT, None, (0, 0, 0)
        )

    if write:
        suffix = "l" if left else "r"
        cv2.imwrite(f"../samples/images/random1/destreched_{suffix}.jpg", img)
    return img


def main():
    """Destrech and stitch panorama image."""
    left = cv2.imread("../samples/images/random1/1l.jpg")
    center = cv2.imread("../samples/images/random1/1c.jpg")
    right = cv2.imread("../samples/images/random1/1r.jpg")

    target_y, target_x, _ = center.shape

    # Undo overlapping of images by cropping 0.5% of the width
    # This could be slightly more precise I guess
    width_l = left.shape[1]
    # Crop at the right
    crop_l = left[:, : width_l - (width_l // 200)]
    destr_l = cv2.resize(crop_l, (target_x, target_y))
    width_r = right.shape[1]
    # Crop at the left
    crop_r = right[:, (width_r // 200) :]
    destr_r = cv2.resize(crop_r, (target_x, target_y))

    # LEFT
    pts_left = np.float32(PTS_LEFT)
    left_image = process_side(destr_l, pts_left)
    # RIGHT
    pts_right = np.float32(PTS_RIGHT)
    right_image = process_side(destr_r, pts_right, left=False)

    # Finally, stitch together
    stitched = cv2.hconcat([left_image, center, right_image])
    cv2.imwrite("../samples/images/random1/stitched.jpg", stitched)


if __name__ == "__main__":
    main()
