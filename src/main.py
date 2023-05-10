import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='Vision-based object detection')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
args = parser.parse_args()

cv2.namedWindow('Video sequence',cv2.WINDOW_NORMAL)

cap=cv2.VideoCapture(args.video_file)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(frame_rate)

while(cap.isOpened()):

    #Got the current frame and pass on to 'frame'
    ret,frame=cap.read()

    #if the current frame cannot be capture, ret=0
    if not ret:
        print("frame missed!")
        break

    # Visualise the input video
    cv2.imshow('Video sequence',frame)

    # The program finishes if the key 'q' is pressed
    key = cv2.waitKey(31)
    if key == ord('q') or key == 27:
        print("Programm finished, mate!")
        break

# Destroy all visualisation windows
cv2.destroyAllWindows()

# Destroy 'VideoCapture' object
cap.release()
    

