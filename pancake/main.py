import argparse
import os
import cv2
import torch

from . import models as m
from . import tracker as tr

from .utils.common import load_data, visualize
from .utils.general import check_img_size, scale_coords
from .utils.torch_utils import time_synchronized
from .utils.parser import get_config

""" CONFIGS """
device = "0"
if not torch.cuda.is_available():
    device = "cpu"


# source = "../samples/streams.txt"
# source = "samples/images/random2_4k/1r-cropped-rotated.jpg"
source = ["../samples/r45/1l", "../samples/r45/1c", "../samples/r45/1r"]


model = "yolov5"
# weights = "yolov5l6.pt"
weights = "../weights/detector/yolov5/yolov5s6_30epochs.pt"

tracker = "deepsort"
tracker_cfg_path = "../configs/tracker/deep_sort.yaml"

img_size = 448
verbose = 2

# visualization
view_img = True
view_img = False
hide_labels = False
hide_conf = False
line_thickness = 2

# detector
conf_thres = 0.55
iou_thres = 0.6
classes = None
agnostic_nms = False


def fix_path(path):
    """Adjust relative path."""
    if type(path) is list:
        return list(map(lambda p: os.path.join(os.path.dirname(__file__), p), path))
    return os.path.join(os.path.dirname(__file__), path)


def main(argv=None, *args, **kwargs):
    """
    LOADING PROCEDURE
    """
    # MODEL SETUP
    weights_cfg = fix_path(weights)
    MODEL = m.MODEL_REGISTRY[model](
        device, weights_cfg, conf_thres, iou_thres, classes, agnostic_nms, img_size
    )

    # TRACKER SETUP
    tracker_cfg = fix_path(tracker_cfg_path)
    tracker_cfg = get_config(config_file=tracker_cfg)
    TRACKER = tr.TRACKER_REGISTRY[tracker](tracker_cfg, device=device)

    # INPUT DATA SETUP
    source_path = fix_path(source)
    DATA, is_webcam = load_data(source_path, MODEL)

    """
    TRACKING PROCEDURE
    """
    for path, img, im0s, vid_cap in DATA:
        # inference
        t1 = time_synchronized()
        pred, prep_img = MODEL.infer(
            img
        )  # pred (tensor): tensor list of detections, on (,6) tensor [xyxy, conf, cls]
        t2 = time_synchronized()

        if verbose > 1:
            print(f"(Inference time: {t2 - t1:.3f}s)")

        # process detections
        for i, det in enumerate(pred):  # iterate over image batch
            if is_webcam or type(source) is list:  # batch_size >= 1
                p, s, im0 = path[i], "%g: " % i, im0s[i].copy()
            else:
                p, s, im0 = path, "", im0s.copy()

            s += f"{prep_img.shape[2:]}, "  # print preprocessed image shape

            # IMPORTANT
            if len(det):
                # Rescale boxes from padded_img_size (padded according to stride) to im0 (original) size
                det[:, :4] = scale_coords(
                    prep_img.shape[2:], det[:, :4], im0.shape
                ).round()

            # tracker update
            tracks = TRACKER.update(
                det, im0
            )  # [x1, y1, x2, y2, centre x, centre y, id]

            # detections per class
            s += "Detections: "
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                s += f"{n} {MODEL._classlabels[int(c)]}{'s' * (n > 1)}, "  # add to string

            # different tracks
            s += f" Tracker-IDs:"
            if len(tracks):
                for track in tracks[:, 6]:
                    s += f" {track}"
                s += ", "

            # print results for current frame
            if verbose > 0:
                print(f"{s} Done.")

            # visualize detections
            if view_img:
                visualize(
                    show_det=True,
                    show_tracks=False,
                    det=det,
                    tracks=tracks,
                    im0=im0,
                    labels=MODEL._classlabels,
                    hide_labels=hide_labels,
                    hide_conf=hide_conf,
                    line_thickness=line_thickness,
                    debug=False,  # Set to true to enable manual stepping
                )
            # input()


if __name__ == "__main__":
    main()
