"""MOSSE (Minimum Output Sum of Squared Error) tracker.

Based on
    David S. Bolme, J. Ross Beveridge, Bruce A. Draper, and Man Lui Yui.
    Visual object tracking using adaptive correlation filters.
    In Conference on Computer Vision and Pattern Recognition (CVPR), 2010.
"""
import cv2

from .tracker import BaseTracker


def centroid(vertices):
    x_list = [vertex for vertex in vertices[::2]]
    y_list = [vertex for vertex in vertices[1::2]]
    x = sum(x_list) // len(x_list)
    y = sum(y_list) // len(y_list)
    return (x, y)


class MosseTracker(BaseTracker):
    def __init__(self, *args, **kwargs) -> None:
        TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create,
        }
        tracker = "mosse"
        self.tracker = TRACKERS[tracker]()
        self.multi = cv2.MultiTracker_create()
        self.multi.addTracker()

    def update(self, det):  # det: list of koordinates x,y , x,y, ...
        self.trackers.update(frame)
        return centroids


def main():
    def write(img, boxes):
        if type(boxes) != list:
            boxes = [boxes]
        for bbox in boxes:
            x0, y0, w, h = bbox
            if type(x0) != int:
                x0, y0, w, h = int(x0), int(y0), int(w), int(h)
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 5)
        cv2.imwrite(f"{time.time()}.jpg", img)

    import glob
    import os
    import time

    import numpy as np
    import torch
    from pancake.detector.detector_yolo_simple import YOLOSimpleDetector

    paths = sorted(
        glob.glob("/home/maurice/git/mauricesvp/pancake/samples/r45/1l/*jpg"),
        key=os.path.basename,
    )
    img = cv2.imread(paths.pop(0))

    # det = YOLOSimpleDetector()
    # detections = det.detect(img)[0]
    # print(detections)

    detections = torch.Tensor(
        [
            [2.84107e03, 7.68151e02, 2.96549e03, 8.37471e02, 5.62032e-01, 2.00000e00],
            [2.73497e03, 7.37642e02, 2.81932e03, 8.01219e02, 3.72773e-01, 2.00000e00],
            [2.55007e03, 8.16756e02, 2.66804e03, 9.11262e02, 3.58595e-01, 2.00000e00],
            [1.86638e03, 8.86099e02, 1.92717e03, 9.33460e02, 3.37813e-01, 2.00000e00],
            [2.11947e03, 8.55039e02, 2.20046e03, 9.09375e02, 3.37788e-01, 2.00000e00],
            [1.57259e03, 8.95265e02, 1.65550e03, 9.68379e02, 2.76119e-01, 2.00000e00],
        ]
    )
    res = []
    for det in detections:
        x0, y0, x1, y1 = np.array(det[:4]).astype("int64")
        res.append([x0, y0, x1 - x0, y1 - y0])

    write(img, res)

    # KCF
    # params = cv2.TrackerKCF_Params()
    # params.max_patch_size = 32000
    # tracker = cv2.TrackerKCF_create(params)

    # CSRT
    params = cv2.TrackerCSRT_Params()
    # params.background_ratio = 1000  # default: 2
    params.use_rgb = True
    params.use_gray = False
    params.psr_threshold = 0.035  # default
    # params.padding = 3
    # params.scale_step = 1.5  # default: 1.02

    # MedianFlow
    # params = cv2.TrackerCSRT_Params()

    trackers = []
    for bbox in res:
        # tracker = cv2.TrackerKCF_create(params)
        tracker = cv2.TrackerCSRT_create(params)
        # tracker = cv2.legacy.TrackerMedianFlow_create()
        # tracker = cv2.legacy.TrackerMOSSE_create()
        # tracker = cv2.legacy.TrackerTLD_create()
        tracker.init(img, bbox)
        trackers.append(tracker)

    for path in paths:
        print(".")
        img = cv2.imread(path)
        boxes = []
        for tracker in trackers:
            success, bbox = tracker.update(img)
            boxes.append(bbox)
        write(img, boxes)
