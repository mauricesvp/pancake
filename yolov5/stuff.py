import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

imgs = ["stitched4k.jpg"]  # This can be an url too

# Inference
results = model(imgs)
# results.show()
results.print()
results.save()  # save to file
a = (results.xyxy[0])


# Draw rectangle around found objects
# We won't need this later
import cv2

img = cv2.imread("stitched4k")

coords = []
for x in a:
    a, b, c, d, *e = x
    cv2.rectangle(img,(int(a), int(b)), (int(c), int(d)), (0,255,0), 1)
cv2.imwrite("stichted4k-labeled", img)
