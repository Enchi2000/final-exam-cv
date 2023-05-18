# ----------------------------------------------------------------

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

h_b_Accumulator = np.zeros((256, 1), dtype=np.float32)
h_g_Accumulator = np.zeros((256, 1), dtype=np.float32)
h_r_Accumulator = np.zeros((256, 1), dtype=np.float32)
rectangles=[]
drawing=False
top_left_pt=None
bottom_right_pt=None   


def plot_histogram(Xinit,Yinit,Xfin,Yfin):
    roi=HSV_FRAME[Yinit:Yfin,Xinit:Xfin]
    hist_r=cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([roi], [2], None, [256], [0, 256])
    return hist_r,hist_g,hist_b

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

    filtered_frame=pool.apply_async(apply_gaussian_filter,[frame])
    RGB_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    HSV_FRAME=cv2.cvtColor(filtered_frame.get(),cv2.COLOR_BGR2HSV_FULL)
    HSV_UNFILTERED=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV_FULL)
    HLS_FRAME=cv2.cvtColor(filtered_frame.get(),cv2.COLOR_BGR2HLS_FULL)
    LUV_FRAME=cv2.cvtColor(filtered_frame.get(),cv2.COLOR_BGR2LUV)
    LAB_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    GRAY_FRAME=cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)



    result=cv2.inRange(RGB_FRAME,(112,141,100),(154,180,141)) #Noise1q
    LUV_SOMBRAS=cv2.inRange(LUV_FRAME,(10,84,121),(49,99,163))
    LUV_SOMBRAS_MENORES=cv2.inRange(LUV_FRAME,(67,84,147),(131,90,160))
    LUV_SOMBRAS_MENORES2=cv2.inRange(LUV_FRAME,(120,86.7,151),(160,93,161))

    RGB_LINEA_ROJA=cv2.inRange(HSV_FRAME,(30,32,171),(51,45,197))

    LINEAS_BLANCAS=cv2.inRange(LAB_FRAME,(214,113,131),(255,125,142))

    result1=cv2.bitwise_or(LUV_SOMBRAS_MENORES2,LUV_SOMBRAS_MENORES)
    result2=cv2.inRange(HSV_FRAME,(52,28,159),(79,69,226)) #Green area

    final=cv2.bitwise_or(result,LUV_SOMBRAS)
    mask=cv2.bitwise_or(final,result2)
    mask=cv2.bitwise_or(mask,result1)
    mask=cv2.bitwise_or(mask,LINEAS_BLANCAS)

    contours,hierarchy  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    final=cv2.bitwise_and(frame,frame,mask=mask)
    detected_objects = []
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<70 and area>20:
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append((x, y, w, h))

    #for x, y, w, h in detected_objects:
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    for rect in rectangles:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 255, 0), thickness=1)
        h_r,h_g,h_b=plot_histogram(rect[0], rect[1], rect[2], rect[3])
        h_r_Accumulator=h_r_Accumulator+h_r
        h_g_Accumulator=h_g_Accumulator+h_g
        h_b_Accumulator=h_b_Accumulator+h_b


    # Visualise the input video
    cv2.imshow('Video sequence',frame)
    cv2.imshow('final',final)
    cv2.imshow('mask',mask)
    cv2.imshow('LINEAROJA',RGB_LINEA_ROJA)
    cv2.imshow('HSV',HSV_UNFILTERED)
    cv2.imshow('HLS',HLS_FRAME)
    cv2.imshow('LUV',LUV_FRAME)
    cv2.imshow('LAB',LAB_FRAME) 
    cv2.imshow('LINEAS_BLANCAS',LINEAS_BLANCAS)


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
plt.plot(h_r_Accumulator,color='red')
plt.plot(h_g_Accumulator,color='green')
plt.plot(h_b_Accumulator,color='blue') 
plt.xlim([0, 256])
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend(['H','S','V'])
plt.show()

end_time = time.time()

total_time = end_time - start_time
print(f'Total time taken: {total_time:.2f} seconds')

    

