import cv2
def scaled_imshow(image, scale=1.0, window_name='Image', waitkey=1):
    """Display an image in a window with scaling."""
    if scale <= 0:
        raise ValueError("Scale must be a positive number.")
    
    height, width = image.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_size)
    
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(waitkey)
