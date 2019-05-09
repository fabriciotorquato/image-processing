# Standard imports
import cv2
import numpy as np
from math import ceil


def write_dots(frame, gray_image, x, y, w, h):
    dado = gray_image[y:y+h, x:x+w]
    cv2.imshow('3', dado)
    keypoints = detector.detect(dado)
    text = str(len(keypoints))
    size = cv2.getTextSize(
        text, font, font_scale, thickness)
    text_width = size[0][0]
    text_height = size[0][1]
    cv2.putText(frame, text, (int(x+(w/2)-(text_width/2)), int(y+(h/2) +
                                                               (text_height/2))), font, font_scale, (0, 0, 255), 2, thickness)
    cv2.polylines(frame, [box],  True,  (0, 123, 123),  3)
    return frame


cap = cv2.VideoCapture('data/dado/dados_2.mp4')
detector = cv2.SimpleBlobDetector_create()
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
thickness = cv2.LINE_AA

if (cap.isOpened() == False):
    print("Error opening video stream or file")

try:
    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret:

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(
                gray_image, 250, 255, cv2.THRESH_BINARY_INV)

            thresh = cv2.medianBlur(thresh, 21)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.imshow('4', thresh)

            if len(contours) > 0:

                for contour in contours:

                    area = cv2.contourArea(contour)
                    if area >= 1000 and area <= 5000:
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        x, y, w, h = cv2.boundingRect(contour)

                        if w > 70 and h > 70:
                            frame = write_dots(
                                frame, gray_image, x, y, ceil(w/2), ceil(h/2))
                            frame = write_dots(
                                frame, gray_image, ceil(
                                    x+w/2), ceil(y+h/2), ceil(w/2), ceil(h/2))

                        elif w > 70:
                            frame=write_dots(
                                frame, gray_image, x, y, ceil(w/2), h)
                            frame=write_dots(
                                frame, gray_image, ceil(x+w/2), y, ceil(w/2), h)

                        elif h > 70:
                            frame=write_dots(
                                frame, gray_image, x, y, w, ceil(h/2))
                            frame=write_dots(
                                frame, gray_image, x, ceil(y+h/2), w, ceil(h/2))
                        else:
                            frame=write_dots(frame, gray_image, x, y, w, h)

            # cv2.imshow('1', gray_image)
            cv2.imshow('2', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
except Exception as ex:
    print(ex)
finally:
    cap.release()
    cv2.destroyAllWindows()
