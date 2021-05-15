import argparse
import cv2

import detector
from utils import load_data, scale_coords, visualize, time_synchronized

""" CONFIGS """
device = "0"

source = "https://www.youtube.com/watch?v=uPvZJWp_ed8&ab_channel=8131okichan"
# source = "samples/images/random2_4k/1r-cropped-rotated.jpg"
# weights = "train_results_yolov5s6/weights/last.pt"
# weights = "yolov5s6.pt"
model = "YOLOv5"
weights = "weights/yolov5s6_100epochs.pt"
img_size = 448
verbose = 2

view_img = True
hide_labels = False
hide_conf = False
line_thickness = 2

conf_thres = 0.65
iou_thres = 0.7
classes = None
agnostic_nms = False


def main(argv=None):
    """
    LOADING PROCEDURE
    """
    # MODEL SETUP
    DETECTOR = detector.MODEL_REGISTRY[model](
        device, 
        weights, 
        conf_thres, 
        iou_thres, 
        classes, 
        agnostic_nms,
        img_size
    )

    # INPUT DATA SETUP
    DATA, is_webcam = load_data(source, DETECTOR)

    """
    TRACKING PROCEDURE
    """
    for path, img, im0s, vid_cap in DATA:
        prep_img = DETECTOR.prep_image_infer(
            img
        )  # prep_img (tensor): resized and padded image preprocessed for inference, 4d tensor [x, R, G, B]

        # inference
        t1 = time_synchronized()
        pred = DETECTOR.infer(
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
                    f"{n} {DETECTOR._classlabels[int(c)]}{'s' * (n > 1)}, "  # add to string
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
                    DETECTOR._classlabels,
                    hide_labels,
                    hide_conf,
                    line_thickness,
                )
            #input()


if __name__ == "__main__":
    main()
