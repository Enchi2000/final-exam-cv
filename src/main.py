# ----------------------------------------------------------------
# 
# copy line to run:
#    'python main.py --video_file  ../../2023_05_05_14_59_37-ball-detection.mp4'
# ----------------------------------------------------------------

import numpy as np
import cv2
import argparse
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Vision-based object detection')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
args = parser.parse_args()


start_time = time.time()

start_point=(617,75)
end_point=(599,683)

# Number of frames in the video
num_frames = 3674

# Calculate the displacement vector
displacement_vector = np.array(end_point) - np.array(start_point)

# Calculate the incremental vector for each frame
incremental_vector = displacement_vector / num_frames

cap=cv2.VideoCapture(args.video_file)

cx=cap.get(3)/2
cy=cap.get(4)/2

f=7500
z=50

mouse_coor=[]
product_list=list()

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Save the coordinates when the left mouse button is clicked
        coor_x,coor_y=x,y

        u=coor_x-cx
        v=cy-coor_y

        x_global=float((u/f)*z)
        y_global=float((-v/f)*z)  
        mouse_coor.append((x_global, y_global))

h_ch1_accumulated = np.zeros((256, 1), dtype=np.float32)
h_ch2_accumulated = np.zeros((256, 1), dtype=np.float32)
h_ch3_accumulated = np.zeros((256, 1), dtype=np.float32)


cv2.namedWindow('Video sequence',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Video sequence', mouse_click)

#Save the drawn rectangles
rectangles=[]

#Command to maintain the drawn rectangles while you keep drawing rectangles
drawing=False

#Initialise array of coordinates of drawn rectangles
top_left_pt=None
bottom_right_pt=None   

#
def plot_histogram(frame_used,Xinit,Yinit,Xfin,Yfin):

    #Changing the region selected to HSV
    roi=frame_used[Yinit:Yfin,Xinit:Xfin]


    hist_channel_1=cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist_channel_2 = cv2.calcHist([roi], [1], None, [256], [0, 256])
    hist_channel_3 = cv2.calcHist([roi], [2], None, [256], [0, 256])

    return hist_channel_1,hist_channel_2,hist_channel_3


def get_cross_product(list_of_object_coordinates):
    mouse_coor=list_of_object_coordinates
    for i in range(1, len(mouse_coor)):
        # Compute the cross product of the two consecutive elements
        cross_product = np.cross(mouse_coor[i - 1], mouse_coor[i])
        product_list.append(cross_product)

    # Convert the array elements to floats and store them in a separate list
    cross_product_values = [float(item) for item in product_list]

    # Check for sign changes in the cross product values
    sign_changes = 0

    for i in range(1, len(cross_product_values)):
        if (cross_product_values[i - 1] >= 0 and cross_product_values[i] < 0) or (cross_product_values[i - 1] <= 0 and cross_product_values[i] > 0):
            # Sign change detected
            sign_changes += 1
    
    return sign_changes,cross_product_values

def get_rectangle(event,x,y,flags,params):
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt
    
    # Check if the left mouse button was pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y
    
        
    # Check if the mouse is being moved while the left button is pressed
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        top_left_pt = (min(x_init, x), min(y_init, y))
        bottom_right_pt = (max(x_init, x), max(y_init, y))
        cv2.rectangle(frame, top_left_pt, bottom_right_pt, color=(0, 255, 0), thickness=1)
        cv2.imshow('Video sequence', frame)
        
    # Check if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangles.append((x_init, y_init, x, y)) 
        
def apply_gaussian_filter(frame,kernel,sigma):
    # Apply a Gaussian filter with a kernel size of 5x5 and sigma value of 1
    filtered_frame = cv2.GaussianBlur(frame, (kernel, kernel), sigma)
    return filtered_frame

def apply_median_filter(frame,kernel):
    filtered_frame=cv2.medianBlur(frame,kernel)
    return filtered_frame


#multiprocess
num_processes = cpu_count()
pool = Pool(num_processes)


while(cap.isOpened()):

    #Got the current frame and pass on to 'frame'
    ret,frame=cap.read()

    #if the current frame cannot be capture, ret=0
    if not ret:
        print("frame missed!")
        break

    # for coord in mouse_coor:
        # cv2.circle(frame, coord, 5, (0, 0, 255), -1)


    # Get the total number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_position = tuple(np.array(start_point) + (num_frames * incremental_vector).astype(int))
    cv2.line(frame, start_point, current_position, (0, 0, 255), 2)

    
    #Applying a filter asyncronous
    filtered_frame=pool.apply_async(apply_gaussian_filter,[frame,5,3])

    #Apply space colors to the video and filtered video.
    RGB_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    HSV_FRAME=cv2.cvtColor(filtered_frame.get(),cv2.COLOR_BGR2HSV_FULL)
    LUV_FRAME=cv2.cvtColor(filtered_frame.get(),cv2.COLOR_BGR2LUV)
    LAB_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    boundaries = np.zeros_like(gray)
    kernel = np.ones((5, 5), np.uint8)

    LAB_SOMBRAS=cv2.inRange(LAB_FRAME,(59,109,133),(100,117,148))

#---------Court area segmented------------------------------
    result2=cv2.inRange(HSV_FRAME,(45,16,117),(101,91,229)) #Green area
    radius = 5  # Radius of the circular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    result2 = cv2.erode(result2, kernel, iterations=1)
#------------------------------------------------------
    contours,hierarchy  = cv2.findContours(result2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    cv2.drawContours(boundaries,contours,-1,(255,255,255),cv2.FILLED)
    boundaries=cv2.bitwise_not(boundaries)
    cv2.imshow('mask1',boundaries)
    red_lines=cv2.bitwise_or(boundaries,result2)
    cv2.imshow("red_lines",red_lines)

#------------------------------------------------------
    #White lines detected (edges)
    kernel = np.ones((5, 5), np.uint8)
    LINEAS_BLANCAS=cv2.inRange(HSV_FRAME,(40,12,203),(75,36,234))
    LINEAS_BLANCAS=cv2.dilate(LINEAS_BLANCAS,kernel,iterations=2)
    mask=cv2.bitwise_or(LINEAS_BLANCAS,red_lines)

#--------------------------------------------------------------------

    mask=cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask=cv2.dilate(mask,kernel,iterations=1)

    contours,hierarchy  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # #array to storage the coordinates
    detected_objects = []

    #Defined area for objects
    for contour in contours:
        area=cv2.contourArea(contour)
        if area>5:
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append((x, y, w, h))

    for x, y, w, h in detected_objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    #creating rectangles by coordinates.
    for rect in rectangles:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 255, 0), thickness=1)

        #Create 3 histograms for each R,G,B space color in the region selected        
        h_1,h_2,h_3=plot_histogram(HSV_FRAME,rect[0], rect[1], rect[2], rect[3])

        #The intensity values of R,G,B accumulated in histograms 
        h_ch1_accumulated=h_ch1_accumulated+h_1
        h_ch2_accumulated=h_ch2_accumulated+h_2
        h_ch3_accumulated=h_ch3_accumulated+h_3


    # Visualise the input video
    cv2.imshow('Video sequence',frame)
    #cv2.imshow('mask',mask)
    # cv2.imshow('mask',mask)
    # cv2.imshow('green_area',result2)
    # cv2.imshow('white lines',LINEAS_BLANCAS)
    #cv2.imshow('LAB_FRAME',LAB_FRAME)
    #cv2.imshow('LUB_FRAME',LUV_FRAME)
    #cv2.imshow('hsv_FRAME',HSV_FRAME)


    # The program finishes if the key 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        print("Programm finished, mate!")
        break

# Destroy all visualisation windows
cv2.destroyAllWindows()

object_crossing, cross_product_values=get_cross_product(mouse_coor)

# Print the cross product values
print(cross_product_values)

# Print the number of sign changes
print('Ball crossed:', object_crossing, ' times')

# Destroy 'VideoCapture' object
cap.release()
# plt.figure(num=1)
# plt.plot(h_ch1_accumulated,color='red')
# plt.plot(h_ch2_accumulated,color='green')
# plt.plot(h_ch3_accumulated,color='blue') 
# plt.xlim([0, 256])
# plt.title('Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.legend(['H','S','V'])
# plt.show()

end_time = time.time()

total_time = end_time - start_time
print(f'Total time taken: {total_time:.2f} seconds')

    
    

