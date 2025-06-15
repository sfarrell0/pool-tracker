import cv2
import numpy as np

def preprocess(img, clahe=False):
    """
    Apply various lighting correction techniques
    """
    # Method 3: Gaussian blur to reduce local variations
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img