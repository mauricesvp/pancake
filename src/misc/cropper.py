import math

import cv2
import imutils
import numpy as np


img = None


def click_event(event, x, y, flags, params):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow("image", img)


def right():
    global img
    filename = "../../samples/images/random2_4k/1r.jpg"
    img = cv2.imread(filename)
    img = imutils.rotate_bound(img, 360-38)
    # Crops

    side = 416 // 2
    start_x = 2900
    end_x = 600
    steps = 7
    def f(x):
        return int((-1682/2335*x)+(8211072/2335))
    for x in range(start_x, end_x, -((start_x-end_x)//steps)):
        y = f(x)
        side = int(1.1*side)
        try:
            cv2.rectangle(img, (x-side, y-side), (x+side, y+side), (0, 0, 255), 5)
        except:
            exit()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.imwrite("FOOO.jpg", img)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def left():
    global img
    filename = "../../samples/images/random2_4k/1l.jpg"
    img = cv2.imread(filename)
    img = imutils.rotate_bound(img, 38)
    # Crops

    side = 450 // 2
    start_x = 1925
    end_x = 4000
    steps = 6
    def f(x):
        return int((310/441*x)+(171316/441))
    for x in range(start_x, end_x, -((start_x-end_x)//steps)):
        y = f(x)
        side = int(1.1*side)
        try:
            cv2.rectangle(img, (x-side, y-side), (x+side, y+side), (0, 0, 255), 5)
        except:
            exit()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.imwrite("FOOO.jpg", img)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    right()
