import cv2
from preprocessing import preprocess
from table_detection import *
from lines import *
from perspectives import four_point_transform
import time

FPS_PRINT = True
SHOW_DEBUG = False
CORNER_PROCESS_INTERVAL = 60

def setup_webcams():
    """
    Sets up the webcam for video capture using OpenCV.
    Returns the VideoCapture object if successful, None otherwise.
    """
    # Open the default webcam
    # '0' refers to the default webcam. 
    cap = cv2.VideoCapture(0)

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
    table_mask, (x, y, w, h), rect = find_largest_black_contour(table_mask, img)
    if SHOW_DEBUG:
        cv2.imshow('Largest Contour', table_mask)
    table_edges = detect_table_edges_from_contour(table_mask, img, min_line_length=199)
    table_corners = find_table_corners_from_edges(table_edges, extension_amount=50, img=img)
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


def mainloop():
    """
    Reads video from the default webcam using OpenCV and displays it in a window.
    Press 'q' to quit the window.
    """
    cap = setup_webcams()[0]
    ctr = 0
    old_time = time.time()
    corners = None
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # If 'ret' is False, it means there was an error reading the frame
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        cv2.imshow('Webcam Feed', frame)
        if ctr % CORNER_PROCESS_INTERVAL == 0:
            new_corners = process_corners(frame)
            if new_corners:
                corners = new_corners
        if corners:
            warped = four_point_transform(frame, corners)
            cv2.imshow('Warped', warped)

        # Wait for 1 millisecond and check if the 'q' key was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ctr += 1
        if FPS_PRINT and ctr % 100 == 0:
            new_time = time.time()
            print(f"FPS: {100/(new_time-old_time):.2f}")
            old_time = new_time
    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    mainloop()