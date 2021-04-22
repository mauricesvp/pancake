import cv2

left = cv2.imread("../samples/images/random1/1l.jpg")
center = cv2.imread("../samples/images/random1/1c.jpg")
right = cv2.imread("../samples/images/random1/1r.jpg")

# Undo overlapping of images by removing 0.5% of the width
width_l = left.shape[1]
crop_l = left[:, :width_l-(width_l//200)]
width_r = right.shape[1]
crop_r = right[:, (width_r//200):]

# TODO: Do de-stretching here

# Simple concat for now
stitched = cv2.hconcat([crop_l, center, crop_r])
cv2.imwrite("../samples/images/random1/stitched.jpg", stitched)
