import cv2
from preprocessing import preprocess
from table_detection import *
from ball_detection import detect_circles
from lines import *
from perspectives import four_point_transform
from motion_detection import detect_motion
import time
from datetime import datetime

FPS_PRINT = True
SHOW_DEBUG = True
CORNER_PROCESS_INTERVAL = 100
FPS_INTERVAL = 100  # Interval for printing FPS
NATIVE_RES = (1280, 720)#(1920, 1080) #(3840, 2160)  # Native resolution of the webcam
PROCESS_RES = NATIVE_RES #(1920, 1080)#(3840, 2160)  # Resolution for the video writer
FOOTAGE_DIR = './footage/'  # Directory to save the footage
GAME_NAME = datetime.now().strftime("game_%Y%m%d_%H%M%S")
FPS = 30.0
EXPOSURE = -5.5
GAIN = 255
BRIGHTNESS = 183
CONTRAST = 176
SATURATION = 128
SHARPENESS = 128


def setup_webcams():
    """
    Sets up the webcam for video capture using OpenCV.
    Returns the VideoCapture object if successful, None otherwise.
    """
    # Open the default webcam
    # '0' refers to the default webcam. 
    cap = cv2.VideoCapture(0) # DSHOW is awful
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_RES[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)  # Set the desired FPS if supported by the webcam
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    cap.set(cv2.CAP_PROP_GAIN, GAIN)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST)
    cap.set(cv2.CAP_PROP_SATURATION, SATURATION)
    cap.set(cv2.CAP_PROP_SHARPNESS, SHARPENESS)



    # Check if the webcam was opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    return [cap]

def process_corners(frame):
    """
    Placeholder for postprocessing the frame.
    Currently, it just returns the frame as is.
    """
    
    img = preprocess(frame)
    if SHOW_DEBUG:
        cv2.imshow('Preprocess OUT', img)
    table_mask = detect_black_table(img)
    if SHOW_DEBUG:
        cv2.imshow('Table Mask', table_mask)
    table_mask = find_largest_black_contour(table_mask, img)
    if SHOW_DEBUG:
        cv2.imshow('Largest Contour', table_mask)
    table_edges = detect_table_edges_from_contour(table_mask, img, min_line_length=199)
    print(f"Table edges: {table_edges}")
    table_corners = find_table_corners_from_edges(table_edges, extension_amount=50, img=img)
    print(f"Table corners: {table_corners}")
    if table_corners is not None:
        corner_img = img.copy()
        # Draw the detected corners on the original frame
        for corner in table_corners:
            cv2.circle(corner_img, corner, 15, (0, 255, 0), -1)
        cv2.imshow('Detected Corners', corner_img)

    if len(table_corners) == 4:
        return table_corners
    else:
        return None

def process_balls(frame):
    detect_circles(frame)
def mainloop():
    """
    Reads video from the default webcam using OpenCV and displays it in a window.
    Press 'q' to quit the window.
    """
    cap = setup_webcams()[0]
    writer = cv2.VideoWriter(f'{FOOTAGE_DIR}/{GAME_NAME}.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         FPS, PROCESS_RES)

    silly_writer = cv2.VideoWriter(f'{FOOTAGE_DIR}/{GAME_NAME}_silly.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         FPS, PROCESS_RES)
    cut_writer = cv2.VideoWriter(f'{FOOTAGE_DIR}/{GAME_NAME}_cut.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         FPS, PROCESS_RES)
    ctr = 0
    old_time = time.time()
    corners = None
    motion_state = {}
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        frame = cv2.resize(frame, PROCESS_RES)

        # If 'ret' is False, it means there was an error reading the frame
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        cv2.imshow('Webcam Feed', frame)
        writer.write(frame)
        motion = detect_motion(frame, motion_state, writer = silly_writer, corners=corners)
        if motion:
            cut_writer.write(frame)

        if ctr-30 % CORNER_PROCESS_INTERVAL == 0:
            new_corners = process_corners(frame)
            if new_corners:
                corners = new_corners
        if corners:
            warped = four_point_transform(frame, corners)
            #cv2.imshow('Warped', warped)
        
        process_balls(frame)

        # Wait for 1 millisecond and check if the 'q' key was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ctr += 1
        if FPS_PRINT and ctr % FPS_INTERVAL == 0:
            new_time = time.time()
            print(f"FPS: {FPS_INTERVAL/(new_time-old_time):.2f}")
            old_time = new_time
    # Release the webcam and destroy all OpenCV windows
    cap.release()
    writer.release()
    silly_writer.release()
    cut_writer.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    mainloop()