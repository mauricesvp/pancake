import argparse

import cv2
import numpy as np

# Constants (or hard coded values)
# YOU MAY HAVE TO CHANGE VALUES HERE IF RESULTS ARE UNEXPECTED

# Calc bt: bt = final_y // BORDER_BTTM_<LEFT/RIGHT>
BORDER_BTTM_LEFT = 2.376
BORDER_BTTM_RIGHT = 2.483

# ROI (Region of Interest)
# These points are relative, so any image resolution will work
PTS_LEFT = [
    [656 / 1280, 288 / 720],
    [1279 / 1280, 208 / 720],
    [1279 / 1280, 432 / 720],
    [537 / 1280, 397 / 720],
]
PTS_RIGHT = [
    [1 / 1280, 215 / 720],
    [925 / 1280, 331 / 720],
    [1006 / 1280, 388 / 720],
    [1 / 1280, 449 / 720],
]


def process_side(img, pts1, path: str, border=True, write=True, left=True):
    """Process one side.

    img: input image
    pts1: ROI (Region of Interest)
    border: Fill border around image to match shape of img
    write: Write to file
    left: Left side of image
    """
    final_y, final_x, _ = img.shape
    # Right side covers a lot more space
    if not left:
        final_x = int(3 * final_x)
    # Max distance of width, height
    # In other words, how large is the smallest rectangle that fits around all 4 points of pts1
    xs = [x[0] for x in pts1]
    ys = [x[1] for x in pts1]
    # mw = int(max(xs) - min(xs))
    # We don't really need a border on the outer side
    mw = final_x
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
        cv2.imwrite(f"{path}destreched_{suffix}.jpg", img)
    return img


def main(path: str):
    """Destrech and stitch panorama image."""
    if not path.endswith("/"):
        path += "/"
    left = cv2.imread(f"{path}1l.jpg")
    center = cv2.imread(f"{path}1c.jpg")
    right = cv2.imread(f"{path}1r.jpg")

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
    pts_left = PTS_LEFT
    for x in pts_left:
        x[0] *= target_x
        x[1] *= target_y
    pts_left = np.float32(pts_left)
    left_image = process_side(destr_l, pts_left, path=path)
    # RIGHT
    pts_right = PTS_RIGHT
    for x in pts_right:
        x[0] *= target_x
        x[1] *= target_y
    pts_right = np.float32(pts_right)
    right_image = process_side(destr_r, pts_right, left=False, path=path)

    # Finally, stitch together
    stitched = cv2.hconcat([left_image, center, right_image])
    cv2.imwrite(f"{path}stitched.jpg", stitched)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, nargs="?", default="../samples/images/random1/"
    )
    args = parser.parse_args()
    main(args.path)
