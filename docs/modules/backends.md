# Backends

_Why is there a need for a backend in the first place? Can we not just feed the detector with the image(s) and be done with it?_

When we first started to think about how to approach the detection,
we thought that it would be sufficient to simply run the detection on the stitched panorama image.
However it quickly became apparent that this was not going to work, as the detections were very poor.
The problem is that the detector is not trained to detect cars that are very small relative to the whole image.

This is why we designed the _DEI_ backend to make sure we can have decent detections.

This however, comes with its own set of problems, mainly
1. Performance overhead
2. Duplication problems

The duplication problems arise, as the individiual subframes (see below) are overlapping (they have to be, else some cars might not get detected at all),
which can cause multiple detections for the same car.
We already have deduplication/merging strategies in play, however they are by no means perfect.
Having more sophisticated strategies in turn would mean an even bigger performance overhead.

The ideal solution would be to have a detector specifically designed and trained for the data at hand, which would make the need of a backend obsolete.

## Basic
  The Basic Backend simply takes the input image(s), and runs the detection on each image.
  It returns the detections as well as a stitched image of all input images.
  (Note that this is currently hard coded to be horizontally stitched)


## DEI (Divide and Conquer)
  The DEI Backend is specifically designed for the detection on the Strasse des 17. Juni,
  using a panorama image (made up by three images).
  
  Because the detections would be very poor if it was run one the panorama directly,
  the Backend first splits the panorama image into partial images (blue squares):
  
  <img src="/gitimg/dei.jpg">
  
  These then get rotated, depending on the proximity to the center (no rotation in the center, more rotation on the outer sides).
  
  This is done as the angle of the cars gets quite skewed on the outer sides, which hinders a successful detection.
  
  The actual detection is now run on the partial images, after which the rotation und splitting are reversed to produce the final results.
