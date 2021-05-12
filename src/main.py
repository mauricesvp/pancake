import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.yolov5_class import Yolov5_Model
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size

# cap = cv2.VideoCapture("samples/Highway - 20090.mp4")
# object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
# object_detector = cv2.createBackgroundSubtractorMOG2(128, cv2.THRESH_BINARY, 1)
# object_detector = cv2.createBackgroundSubtractorKNN(history=10, detectShadows=False)

""" CONFIGS """
device = 'cpu'

source = 'samples/images/random2_4k/1c.jpg'
weights = 'train_results_yolov5s6/weights/last.pt'
view_img = True
img_size = 416

conf_thres = 0.3
iou_thres = 0.5
classes = None
agnostic_nms = False

def load_data(source: str, img_size: int):
    """
    :param source (str): data source (webcam, image, video, directory, glob, youtube video, HTTP stream)
    :param img_size (int): inference size (pixels)
    """
    is_webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
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
    YOLO._init_infer(check_img_size(img_size, s=YOLO._stride)) 

    # INPUT DATA SETUP
    DATA = load_data(source)

    """
    TRACKING PROCEDURE
    """
    for path, img, im0s, vid_cap in DATA:
        pred = YOLO.infer(img)
        print(pred)






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