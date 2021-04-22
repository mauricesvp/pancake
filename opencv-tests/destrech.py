import cv2
import numpy as np


def main():
    left = cv2.imread("../samples/images/random1/1l.jpg")
    center = cv2.imread("../samples/images/random1/1c.jpg")
    right = cv2.imread("../samples/images/random1/1r.jpg")

    # Undo overlapping of images by cropping 0.5% of the width
    # This could be slightly more precise I guess
    width_l = left.shape[1]
    # Crop at the right
    crop_l = left[:, :width_l-(width_l//200)]
    width_r = right.shape[1]
    # Crop at the left
    crop_r = right[:, (width_r//200):]

    pts1 = [
            [630,175],
            [1279,1],
            [1279,719],
            [248,399],
    ]

    # Points in the image
    pts1 = [
            [656,288],
            [1279,208],
            [1279,432],
            [537,397],
    ]

    pts2 = [
            [0,0],
            [1280,0],
            [1280,720],
            [0,720],
    ]
    # Sort of preserve ratio
    mh = 432-208
    mw = 1279-537
    # Where points should end up
    pts2 = [
            [0,0],
            [mw,0],
            [mw,mh],
            [0,mh],
    ]

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    destr_l = cv2.resize(crop_l, (1280,720))
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(destr_l, M, (mw, mh))

    bt = 303
    tp = 720 - bt - mh
    left = (1280 - mw)
    # Our points slice is smaller than image, fill it up with border (so that concat works still)
    dst = cv2.copyMakeBorder(warped, tp, bt, left, 0, cv2.BORDER_CONSTANT, None, (0,0,0))

    cv2.imwrite("../samples/images/random1/destreched2.jpg", dst)

    stitched = cv2.hconcat([dst, center, crop_r])
    cv2.imwrite("../samples/images/random1/stitched.jpg", stitched)


if __name__ == "__main__":
    main()
