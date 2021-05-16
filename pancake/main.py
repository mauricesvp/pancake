import argparse

import cv2

import models as m
import tracker as tr

from utils.common import load_data, visualize
from utils.general import check_img_size, scale_coords
from utils.torch_utils import time_synchronized
from utils.parser import get_config

""" CONFIGS """
device = "0"

source = "https://www.youtube.com/watch?v=uPvZJWp_ed8&ab_channel=8131okichan"
# source = "samples/images/random2_4k/1r-cropped-rotated.jpg"
# weights = "train_results_yolov5s6/weights/last.pt"
# weights = "yolov5s6.pt"
model = "yolov5"
weights = "weights/detector/yolov5/yolov5s6_30epochs.pt"

tracker = "deepsort"
cfg = get_config(config_file="configs/tracker/deep_sort.yaml")

img_size = 448
verbose = 2

# visualization
view_img = True
hide_labels = False
hide_conf = False
line_thickness = 2

# detector
conf_thres = 0.65
iou_thres = 0.7
classes = None
agnostic_nms = False


def main(argv=None):
    """
    LOADING PROCEDURE
    """
    # MODEL SETUP
    MODEL = m.MODEL_REGISTRY[model](
        device, 
        weights, 
        conf_thres, 
        iou_thres, 
        classes, 
        agnostic_nms,
        img_size
        )

    # TRACKER SETUP
    TRACKER = tr.TRACKER_REGISTRY[tracker](
        cfg,
        device=device
    )

    # INPUT DATA SETUP
    DATA, is_webcam = load_data(
        source, 
        MODEL
    )

    """
    TRACKING PROCEDURE
    """
    for path, img, im0s, vid_cap in DATA:
        prep_img = MODEL.prep_image_infer(
            img
        )  # prep_img (tensor): resized and padded image preprocessed for inference, 4d tensor [x, R, G, B]

        # inference
        t1 = time_synchronized()
        pred = MODEL.infer(
            prep_img
        )  # pred (tensor): tensor list of detections, on (,6) tensor [xyxy, conf, cls]
        t2 = time_synchronized()

        if verbose > 1:
            print(f"(Inference time: {t2 - t1:.3f}s)")

        # process detections
        for i, det in enumerate(pred):  # iterate over image batch
            if is_webcam:  # batch_size >= 1
                p, s, im0 = path[i], "%g: " % i, im0s[i].copy()
            else:
                p, s, im0 = path, "", im0s.copy()

            s += f"{prep_img.shape[2:]} || "  # print preprocessed image shape

            # IMPORTANT
            if len(det):
                # Rescale boxes from padded_img_size (padded according to stride) to im0 (original) size
                det[:, :4] = scale_coords(
                    prep_img.shape[2:], det[:, :4], im0.shape
                ).round()

            # detections per class
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                s += (
                    f"{n} {MODEL._classlabels[int(c)]}{'s' * (n > 1)}, "  # add to string
                )

            TRACKER.update(
                det,
                im0
            )

            # print results for current frame
            if verbose > 0:
                print(f"{s} Done.")

            # visualize detections
            if view_img:
                visualize(
                    det,
                    p,
                    im0,
                    MODEL._classlabels,
                    hide_labels,
                    hide_conf,
                    line_thickness,
                )
            #input()


if __name__ == "__main__":
    main()
