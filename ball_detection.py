import cv2
import numpy as np
def detect_circles(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Grayscale', grayscale)