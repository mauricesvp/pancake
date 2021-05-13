import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.yolov5_class import Yolov5_Model
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, scale_coords
from utils.plots import colors, plot_one_box

# cap = cv2.VideoCapture("samples/Highway - 20090.mp4")
# object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
# object_detector = cv2.createBackgroundSubtractorMOG2(128, cv2.THRESH_BINARY, 1)
# object_detector = cv2.createBackgroundSubtractorKNN(history=10, detectShadows=False)

""" CONFIGS """
device = 'cpu'

source = 'samples/c1_sample.avi'
weights = 'train_results_yolov5s6/weights/last.pt'
img_size = 3840

view_img = True
hide_labels = False
hide_conf = False
line_thickness = 3

conf_thres = 0.3
iou_thres = 0.5
classes = None
agnostic_nms = False

def load_data(source: str, img_size: int, is_webcam: bool):
    """
    :param source (str): data source (webcam, image, video, directory, glob, youtube video, HTTP stream)
    :param img_size (int): inference size (pixels)
    """
    if is_webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        return LoadWebcam(source, img_size=img_size, stride=YOLO._stride)
    else:
        return LoadImages(source, img_size=img_size, stride=YOLO._stride)


if __name__ == '__main__':
    """ 
    LOADING PROCEDURE
    """
    # MODEL SETUP
    YOLO = Yolov5_Model(device, weights, conf_thres, iou_thres, classes, agnostic_nms)
    padded_img_size = check_img_size(img_size, s=YOLO._stride)
    YOLO._init_infer(padded_img_size) 

    # INPUT DATA SETUP
    is_webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    DATA = load_data(source, padded_img_size, is_webcam)

    """
    TRACKING PROCEDURE
    """
    for path, img, im0s, vid_cap in DATA:
        pred, img = YOLO.infer(img)
        
        for i, det in enumerate(pred):
            if is_webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), DATA.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(DATA, 'frame', 0)

            s += f'{img.shape[2:]}%{img.shape[2:]}'  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size (padded according to stride) to im0 (original) size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {YOLO._names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (YOLO._names[c] if hide_conf else f'{YOLO._names[c]} {conf:.2f}')

                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

            print(f'{s}Done')
            
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


    # while True:
    #     # Read one frame
    #     ret, frame = cap.read()
    #     roi = frame[400:720, 300:700]

    #     mask = object_detector.apply(roi)
    #     _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #     for cnt in contours:
    #         # Calculate area and remove small elements
    #         area = cv2.contourArea(cnt)
    #         if area > 100:
    #             cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)

    #     cv2.imshow("Mask", mask)
    #     cv2.imshow("Frame", frame)
    #     cv2.imshow("roi", roi)

    #     key = cv2.waitKey(30)
    #     if key == 27:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()