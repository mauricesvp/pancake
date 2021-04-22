"""Open image, show and print x,y coordinates of mouse clicks."""
import argparse
import os

import cv2


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow("image", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, nargs="?", default="")
    args = parser.parse_args()
    filename = ""
    if not args.file:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../samples/images/random1/1r.jpg")
    else:
        filename = args.file
    img = cv2.imread(filename)
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_event)
    print("Draw points using left mouse click.")
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
