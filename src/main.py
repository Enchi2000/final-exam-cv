import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Segmentation of players and ball')
parser.add_argument('--video_file', type=str, default='camera', help='Video file used for segmentation')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_file)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame missed!")
        break

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper thresholds for court and shadows (adjust these values as per your video)
    court_lower = np.array([35, 50, 50], dtype=np.uint8)
    court_upper = np.array([85, 255, 255], dtype=np.uint8)

    shadows_lower = np.array([0, 0, 0], dtype=np.uint8)
    shadows_upper = np.array([179, 50, 70], dtype=np.uint8)

    players_lower = np.array([0, 0, 0], dtype=np.uint8)
    players_upper = np.array([179, 255, 255], dtype=np.uint8)

    ball_lower = np.array([0, 0, 0], dtype=np.uint8)
    ball_upper = np.array([179, 255, 255], dtype=np.uint8)

   # Create a mask for players using the specified thresholds
    players_mask = cv2.inRange(hsv_frame, players_lower, players_upper)

    # Create a mask for the ball using the specified thresholds
    ball_mask = cv2.inRange(hsv_frame, ball_lower, ball_upper)

    # Combine the players mask and the ball mask using bitwise OR operation
    combined_mask = cv2.bitwise_or(players_mask, ball_mask)

    # Apply the combined mask to the frame to extract the players and the ball
    segmented_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Display the segmented frame showing both players and the ball
    cv2.imshow('Segmented Frame', segmented_frame)

    cv2.imshow('Mask Frame', combined_mask)

    
    cv2.waitKey(31)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
