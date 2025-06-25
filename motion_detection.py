import cv2
import numpy as np
BLUR_KERNEL_SIZE = (15, 15)
MOTION_CONSTANT = 30.0
MOTION_THRESHOLD = 700
def detect_motion(frame, motion_state, writer=None, corners=[]):
    
    blurred = cv2.GaussianBlur(frame, BLUR_KERNEL_SIZE, 0)
    gray_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray_frame = np.float32(gray_frame)
    bg_frame = motion_state.get('bg_frame')
    if bg_frame is None:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_state['bg_frame'] = np.float32(grayscale)
        return False
    else:
        motion_state['bg_frame'] = cv2.accumulateWeighted(gray_frame, bg_frame, 1.0/MOTION_CONSTANT)
        if writer is not None:
            writer.write(cv2.cvtColor(motion_state['bg_frame'].astype(np.uint8), cv2.COLOR_GRAY2BGR))

        diff_frame = cv2.absdiff(bg_frame, gray_frame)
        _, thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
        # mask out corners
        if corners:
            min_x = min(c[0] for c in corners)
            max_x = max(c[0] for c in corners)
            min_y = min(c[1] for c in corners)
            max_y = max(c[1] for c in corners)
            mask = np.zeros_like(thresh_frame)
            cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, -1)
            thresh_frame = cv2.bitwise_and(thresh_frame, mask)
        
        
        # CONTOURS??
        motion = cv2.countNonZero(thresh_frame)
        thresh_frame = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(thresh_frame, f'Motion: {motion}', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2)
        #cv2.imshow('Threshold Frame', thresh_frame)
        #cv2.imshow('Background Frame', np.uint8(bg_frame))
        return motion > MOTION_THRESHOLD
