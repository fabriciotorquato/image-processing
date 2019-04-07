# Standard imports
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('dados.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # binary image
        _, thresh = cv2.threshold(gray_image, 247, 247, cv2.THRESH_BINARY)

        # get contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = cv2.LINE_AA

        for contour in contours:

            area = cv2.contourArea(contour)
            if area >= 1000 and area <= 5000:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                x, y, w, h = cv2.boundingRect(contour)
                dado = gray_image[y:y+h, x:x+w]

                # Detect blobs.
                keypoints = detector.detect(dado)

                # Get text siza
                text = str(len(keypoints))
                size = cv2.getTextSize(text, font, font_scale, thickness)
                text_width = size[0][0]
                text_height = size[0][1]
                cv2.putText(frame, text, (int(x+(w/2)-(text_width/2)), int(y+(h/2) +
                                                                           (text_height/2))), font, font_scale, (0, 0, 255), 2, thickness)
                cv2.polylines(frame, [box],  True,  (0, 123, 123),  3)

            # Display the resulting frame
            cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break   

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
