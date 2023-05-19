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

start_time = time.time()

start_point=(617,75)
end_point=(599,683)

# Number of frames in the video
num_frames = 3674

# Calculate the displacement vector
displacement_vector = np.array(end_point) - np.array(start_point)

# Calculate the incremental vector for each frame
incremental_vector = displacement_vector / num_frames


h_ch1_accumulated = np.zeros((256, 1), dtype=np.float32)
h_ch2_accumulated = np.zeros((256, 1), dtype=np.float32)
h_ch3_accumulated = np.zeros((256, 1), dtype=np.float32)

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
        
def apply_gaussian_filter(frame):
    # Apply a Gaussian filter with a kernel size of 5x5 and sigma value of 1
    filtered_frame = cv2.GaussianBlur(frame, (7, 7), 5)
    return filtered_frame

def apply_median_filter(frame):
    filtered_frame=cv2.medianBlur(frame,3)
    return filtered_frame


parser = argparse.ArgumentParser(description='Vision-based object detection')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
args = parser.parse_args()

cv2.namedWindow('Video sequence',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Video sequence', get_rectangle)

#multiprocess
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

    # Get the total number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    current_position = tuple(np.array(start_point) + (num_frames * incremental_vector).astype(int))
    cv2.line(frame, start_point, current_position, (0, 0, 255), 2)

    

    #Applying a filter asyncronous
    filtered_frame=pool.apply_async(apply_gaussian_filter,[frame])

    #Apply space colors to the video and filtered video.
    RGB_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    HSV_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV_FULL)
    HLS_FRAME=cv2.cvtColor(filtered_frame.get(),cv2.COLOR_BGR2HLS_FULL)
    LUV_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGR2LUV)
    LAB_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    GRAY_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)

    #RGB ranges for area close to the goal
    result=cv2.inRange(RGB_FRAME,(112,141,100),(154,180,141)) #Noise1q

    #LUV ranges defined
    LUV_SOMBRAS_LINEAS_BLANCAS=cv2.inRange(LUV_FRAME,(80,90,128),(120,96,138))
    # cv2.imshow('LINEAS_BLANCAS_SOMBRAS',LUV_SOMBRAS_LINEAS_BLANCAS)

    LUV_SOMBRAS=cv2.inRange(LUV_FRAME,(10,84,121),(49,99,163))
    #Lamp light only Shadows reflected 
    LAB_SOMBRAS_MENORES=cv2.inRange(LAB_FRAME,(50,108,131),(125,128,149))
    # cv2.imshow('LAB_SOMBRAS_PELOTAS',LAB_SOMBRAS_MENORES)


    #White lines detected (edges)
    LINEAS_BLANCAS=cv2.inRange(LAB_FRAME,(214,113,131),(255,125,142))
    # cv2.imshow('Lineas blancas',LINEAS_BLANCAS)

    #Court area segmented
    result2=cv2.inRange(HSV_FRAME,(40,19,100),(100,95,226)) #Green area

    #Court fence and goal region white
    final=cv2.bitwise_or(result,LUV_SOMBRAS)

    Inside_Boundaries=cv2.bitwise_or(result2,LUV_SOMBRAS)
    # cv2.imshow('inside boundaries',Inside_Boundaries)

    SHADOW_PERSONS=cv2.inRange(LAB_FRAME,(32,109,131),(110,129,150))
    # cv2.imshow('SHADOW_PERSONS',SHADOW_PERSONS)

    mask_fence_goal=cv2.bitwise_or(final,result2)
    mask_shadows_fence_goal=cv2.bitwise_or(mask_fence_goal,LAB_SOMBRAS_MENORES)
    mask_shadows_persons=cv2.bitwise_or(mask_shadows_fence_goal,SHADOW_PERSONS)
    #Mask of fence, shadows, goal region and white lines
    mask=cv2.bitwise_or(mask_shadows_fence_goal,LINEAS_BLANCAS)
    mask=cv2.bitwise_or(mask,LUV_SOMBRAS_LINEAS_BLANCAS)
    
    final_mask=pool.apply_async(apply_median_filter,[mask])
    contours,hierarchy  = cv2.findContours(final_mask.get(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('final_mask',final_mask.get())
    #Applying mask filter
    final=cv2.bitwise_and(frame,frame,mask=final_mask.get())
    
    #array to storage the coordinates
    detected_objects = []

    #Defined area for objects
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<70 and area>5:
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append((x, y, w, h))

    for x, y, w, h in detected_objects:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    #creating rectangles by coordinates.
    for rect in rectangles:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 255, 0), thickness=1)

        #Create 3 histograms for each R,G,B space color in the region selected        
        h_1,h_2,h_3=plot_histogram(LUV_FRAME,rect[0], rect[1], rect[2], rect[3])

        #The intensity values of R,G,B accumulated in histograms 
        h_ch1_accumulated=h_ch1_accumulated+h_1
        h_ch2_accumulated=h_ch2_accumulated+h_2
        h_ch3_accumulated=h_ch3_accumulated+h_3


    # Visualise the input video
    cv2.imshow('Video sequence',frame)
    # cv2.imshow('HSV',HSV_FRAME)
    # cv2.imshow('mask',mask)
    # cv2.imshow('HSV_boundaries',result2)
    # cv2.imshow('HLS',HLS_FRAME)
    # cv2.imshow('LUV_SOMBRAS',LUV_FRAME)
    # cv2.imshow('LAB',LAB_FRAME) 
    # cv2.imshow('LINEAS_BLANCAS',LINEAS_BLANCAS)


    # The program finishes if the key 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        print("Programm finished, mate!")
        break

# Destroy all visualisation windows
cv2.destroyAllWindows()

# Destroy 'VideoCapture' object
cap.release()
plt.figure(num=1)
plt.plot(h_ch1_accumulated,color='red')
plt.plot(h_ch2_accumulated,color='green')
plt.plot(h_ch3_accumulated,color='blue') 
plt.xlim([0, 256])
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
# plt.legend(['H','S','V'])
plt.legend(['L','U','V'])
plt.show()

end_time = time.time()

total_time = end_time - start_time
print(f'Total time taken: {total_time:.2f} seconds')

    

