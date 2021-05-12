import time
from multiprocessing import Pool

import cv2

cap = cv2.VideoCapture("https://media.dcaiti.tu-berlin.de/tccams/1c/axis-cgi/mjpg/video.cgi?camera=1&rotation=0&audio=0&mirror=0&fps=0&compression=50")
capl = cv2.VideoCapture("https://media.dcaiti.tu-berlin.de/tccams/1l/axis-cgi/mjpg/video.cgi?camera=1&rotation=0&audio=0&mirror=0&fps=0&compression=50")
capr = cv2.VideoCapture("https://media.dcaiti.tu-berlin.de/tccams/1r/axis-cgi/mjpg/video.cgi?camera=1&rotation=0&audio=0&mirror=0&fps=0&compression=50")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 3.0, (3840,2160))


def save_video():
    try:
        while(cap.isOpened()):
            ret, frame = cap.read()
            out.write(frame)
    except KeyboardInterrupt:
        pass


def grab(pos: str):
    try:
        capt = None
        if pos == "l":
            capt = capl
        if pos == "c":
            capt = cap
        if pos == "r":
            capt = capr
        while True:
            ret, frame = capt.read()
            cv2.imwrite(f"testing/{pos}{time.time():.2f}.jpg", frame)
            # frames.append((f"{time.time():.2f}.jpg", frame))
    except KeyboardInterrupt:
        # for frame in frames:
            # cv2.imwrite(f"testing/{frame[0]}", frame[1])
        pass


def main():
    frames = []
    with Pool() as pool:
        a = ["l", "c", "r"]
        pool.map(grab, a)



if __name__ == "__main__":
    print("Started stream script")
    main()
    # save_video()
