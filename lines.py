import cv2
import numpy as np

def calculate_line_length(line):
    """Calculate the length of a line segment"""
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_line_angle(line):
    """Calculate the angle of a line in degrees"""
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle

