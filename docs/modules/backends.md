# Backends

## Basic
  The Basic Backend simply takes the input image(s), and runs the detection on each image.
  It returns the detections as well as a stitched image of all input images.
  (Note that this is currently hard coded to be horizontally stitched)


## DEI (Divide and Conquer)
  The DEI Backend is specifically designed for the detection on the Strasse des 17. Juni,
  using a panorama image (made up by three images).
  Because the detections would be very poor if it was run one the panorama directly,
  the Backend first splits the panorama image into partial images.
  These then get rotated, depending on the proximity to the center (no rotation in the center, more rotation on the outer sides).
  This is done as the angle of the cars gets quite skewed on the outer sides, which hinders a successful detection.
  The actual detection is now run on the partial images, after which the rotation und splitting are reversed to produce the final results.
