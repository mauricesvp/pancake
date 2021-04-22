import cv2

cap = cv2.VideoCapture("../samples/Highway - 20090.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
object_detector = cv2.createBackgroundSubtractorMOG2(128,cv2.THRESH_BINARY,1)
# object_detector = cv2.createBackgroundSubtractorKNN(history=10, detectShadows=False)

while True:
    # Read one frame
    ret, frame = cap.read()
    roi = frame[400: 720, 300: 700]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    cv2.imshow("roi", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
