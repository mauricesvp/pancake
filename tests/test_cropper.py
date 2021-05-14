from pancake.misc.cropper import partial

partial("../samples/images/random2_4k/1l.jpg", side="l", imwrite=True, imwrite_filename="left.jpg")
partial("../samples/images/random2_4k/1r.jpg", side="r", imwrite=True, imwrite_filename="right.jpg")
