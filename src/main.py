import numpy as np
import cv2
import argparse
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt

start_time = time.time()

parser = argparse.ArgumentParser(description='Vision-based object detection')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
args = parser.parse_args() 
    
def apply_gaussian_filter(frame):
    # Apply a Gaussian filter with a kernel size of 5x5 and sigma value of 1
    filtered_frame = cv2.GaussianBlur(frame, (5, 5), 1)
    return filtered_frame

cv2.namedWindow('Video sequence',cv2.WINDOW_NORMAL)
cv2.namedWindow('Filtered Frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV mask',cv2.WINDOW_NORMAL)

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

    HSV_frame=cv2.cvtColor(filtered_frame.get(),cv2.COLOR_BGR2HSV)
    
    mask=cv2.inRange(HSV_frame,(42,38,164),(57,67,211))
    mask=cv2.bitwise_not(mask)
    
    result=cv2.bitwise_and(frame, frame, mask=mask)

    # Display the filtered frame
    cv2.imshow('Filtered Frame', filtered_frame.get())

    # Visualise the input video
    cv2.imshow('Video sequence',frame)

    cv2.imshow('HSV mask',result)

    # The program finishes if the key 'q' is pressed
    key = cv2.waitKey(31)
    if key == ord('q') or key == 27:
        print("Programm finished, mate!")
        break

# Destroy all visualisation windows
cv2.destroyAllWindows()

# Destroy 'VideoCapture' object
cap.release()

end_time = time.time()

total_time = end_time - start_time
print(f'Total time taken: {total_time:.2f} seconds')

    

