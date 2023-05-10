import numpy as np
import cv2
import argparse
from multiprocessing import Pool, cpu_count

def apply_gaussian_filter(frame):
    # Apply a Gaussian filter with a kernel size of 5x5 and sigma value of 0
    filtered_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return filtered_frame

parser = argparse.ArgumentParser(description='Vision-based object detection')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
args = parser.parse_args()

cv2.namedWindow('Video sequence',cv2.WINDOW_NORMAL)

num_processes = cpu_count()
pool = Pool(num_processes)

cap=cv2.VideoCapture(args.video_file)

while(cap.isOpened()):

    #Got the current frame and pass on to 'frame'
    ret,frame=cap.read()

    #if the current frame cannot be capture, ret=0
    if not ret:
        print("frame missed!")
        break

    filtered_frame = pool.apply_async(apply_gaussian_filter, [frame])

    # Display the filtered frame
    cv2.imshow('Filtered Frame', filtered_frame.get())

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
    

