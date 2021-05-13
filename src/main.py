import argparse
import cv2

from models.yolov5_class import Yolov5Model
from utils.common import load_data, visualize
from utils.general import check_img_size, scale_coords
from utils.torch_utils import time_synchronized

""" CONFIGS """
device = '0'

source = 'https://www.youtube.com/watch?v=uPvZJWp_ed8&ab_channel=8131okichan'
#source = 'samples/c1_sample.avi'
weights = 'train_results_yolov5s6/weights/last.pt'
img_size = 416
verbose = 2

view_img = True
hide_labels = False
hide_conf = False
line_thickness = 2

conf_thres = 0.65
iou_thres = 0.7
classes = None
agnostic_nms = False


if __name__ == '__main__':
    """ 
    LOADING PROCEDURE
    """
    # MODEL SETUP
    YOLO = Yolov5Model(device, weights, conf_thres, iou_thres, classes, agnostic_nms)
    padded_img_size = check_img_size(img_size, s=YOLO._stride)
    YOLO._init_infer(padded_img_size) 
    
    # INPUT DATA SETUP
    is_webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    DATA = load_data(source, YOLO, padded_img_size, is_webcam)

    """
    TRACKING PROCEDURE
    """
    for path, img, im0s, vid_cap in DATA:
        prep_img = YOLO.prep_image_infer(img) # prep_img (tensor): resized and padded image preprocessed for inference, 4d tensor [x, R, G, B]

        # inference
        t1 = time_synchronized()
        pred = YOLO.infer(prep_img) # pred (tensor): tensor list of detections, on (,6) tensor [xyxy, conf, cls]
        t2 = time_synchronized()

        if verbose > 1:
            print(f'(Inference time: {t2 - t1:.3f}s)')

        # process detections 
        for i, det in enumerate(pred): # iterate over all inferenced images
            if is_webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s.copy()

            s += f'{prep_img.shape[2:]}'  # print preprocessed image shape

            # IMPORTANT
            if len(det):
                # Rescale boxes from img_size (padded according to stride) to im0 (original) size
                det[:, :4] = scale_coords(prep_img.shape[2:], det[:, :4], im0.shape).round()

            # detections per class
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                s += f"{n} {YOLO._classlabels[int(c)]}{'s' * (n > 1)}, "  # add to string

            # print results for current frame
            if verbose > 0:
                print(f'{s} Done.')

            # visualize detections
            if view_img:
                visualize(det, p, im0, YOLO._classlabels, hide_labels, hide_conf, line_thickness)
        