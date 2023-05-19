""" stds-sample-code-for-object-detection.py


    USAGE: 
    $ python main.py --video_file ../../2023_05_05_14_59_37-ball-detection.mp4 &

"""

# -----------------------  Import standard libraries ----------------------- #
from __future__ import print_function
import cv2 
import argparse


# ------------------------- Define utility functions ----------------------- #
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)


# ------------------ Define and initialise variables ---------------------- #
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Input video'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


# ------------------ Parse data from the command line terminal ------------- #
parser = argparse.ArgumentParser(description='Vision-based object detection')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
# parser.add_argument('--frame_resize_percentage', type=float )
args = parser.parse_args()

# ------------------ Read video sequence file ------------------------------ #
cap = cv2.VideoCapture(args.video_file)

# ------------- Create two new windows for visualisation purposes ---------- #
cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)

# ------------ Configure trackbars for the low and hight HSV values -------- #
cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)


def draw_rectangle(contour):
    #Getting the coordinates for the rectangle
    perimeter=cv2.arcLength(contour,True)
    polynomio=cv2.approxPolyDP(contour,0.02*perimeter,True)
    x_,y_,w,h=cv2.boundingRect(polynomio)
    
    return x_,y_,w,h

# ------- Main loop --------------- #
while True:

    # Read current frame
    ret, frame = cap.read()

    # Check if the image was correctly captured
    if frame is None:
        break

    # # Resize current frame
    # width = int(frame.shape[1] * args.frame_resize_percentage / 100)
    # height = int(frame.shape[0] * args.frame_resize_percentage / 100)
    # dim = (width, height)
    # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Apply the median filter
    frame = cv2.medianBlur(frame,5)

    # Convert the current frame from BGR to HSV
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply a threshold to the HSV image
    frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    # Convert the current frame from BGR to  
    framem = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Filter out the grassy region from current frame and keep the moving object only
    bitwise_AND = cv2.bitwise_and(frame, frame, mask=frame_threshold)

    # Filter out the grassy re+gion from current frame and keep the moving object only
    bitwise_AND = cv2.bitwise_and(frame, frame, mask=frame_threshold)
    contours, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        #finding the area of the contour detected
        area=cv2.contourArea(contour)
        print(area)
        #declaring a mininum of an area to draw the rectangle
        if area>2 and area<8:  
        # #calling a function to determine the correct polynomio of the detected object and drawing the rectangle
            x_,y_,w,h=draw_rectangle(contour)
            cv2.rectangle(bitwise_AND,(x_-5,y_-5),(x_+20,y_+20),(0,255,0),1)
# 

    # Visualise both the input video and the object detection windows
    cv2.imshow(window_capture_name, frame)
    cv2.imshow(window_detection_name, bitwise_AND)

    

    cv2.waitKey(25)

    # The program finishes if the key 'q' is pressed
    key = cv2.waitKey(5)
    if key == ord('q') or key == 27:
        print("Programm finished, mate!")
        break

# Destroy all visualisation windows
cv2.destroyAllWindows()

# Destroy 'VideoCapture' object
cap.release()