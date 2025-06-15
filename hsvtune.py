import cv2
import numpy as np

def scaled_imshow(image, scale=1.0, window_name='Image', waitkey=10):
    """
    Display an image in a window with scaling.
    
    :param image: The image to display.
    :param scale: Scaling factor for the image.
    :param window_name: Name of the window where the image will be displayed.
    """
    if scale <= 0:
        raise ValueError("Scale must be a positive number.")
    
    # Resize the image
    height, width = image.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_size)
    
    # Show the resized image
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(waitkey)
def preprocess_image_for_lighting(img):
    """
    Apply various lighting correction techniques
    """
    # Method 1: Histogram Equalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])  # Equalize L channel
    equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    clahe_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Method 3: Gaussian blur to reduce local variations
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    return equalized, clahe_result, blurred

def create_hsv_tuner(image_path):
    """
    Interactive HSV tuner with trackbars to find the perfect color range
    """
    # Load image
    img = cv2.imread(image_path)
    equalized, clahe_result, blurred = preprocess_image_for_lighting(img)
    img = clahe_result  # Use the CLAHE result for better lighting correction
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create window and trackbars
    cv2.namedWindow('HSV Tuner')
    cv2.namedWindow('Original')
    cv2.namedWindow('Mask')
    
    # Create trackbars for HSV ranges
    # Hue: 0-179 (OpenCV uses 0-179 instead of 0-359)
    cv2.createTrackbar('H Min', 'HSV Tuner', 40, 179, lambda x: None)
    cv2.createTrackbar('H Max', 'HSV Tuner', 80, 179, lambda x: None)
    
    # Saturation: 0-255
    cv2.createTrackbar('S Min', 'HSV Tuner', 20, 255, lambda x: None)
    cv2.createTrackbar('S Max', 'HSV Tuner', 255, 255, lambda x: None)
    
    # Value: 0-255
    cv2.createTrackbar('V Min', 'HSV Tuner', 20, 255, lambda x: None)
    cv2.createTrackbar('V Max', 'HSV Tuner', 100, 255, lambda x: None)
    
    print("HSV Color Tuning Guide:")
    print("=" * 50)
    print("HUE (H): Controls the color type")
    print("  0-10: Red")
    print("  10-25: Orange") 
    print("  25-35: Yellow")
    print("  35-85: Green")
    print("  85-125: Blue")
    print("  125-165: Purple/Violet")
    print("  165-179: Pink/Magenta")
    print()
    print("SATURATION (S): Controls color intensity")
    print("  0: Grayscale (no color)")
    print("  255: Fully saturated (vivid color)")
    print()
    print("VALUE (V): Controls brightness")
    print("  0: Black")
    print("  255: Bright")
    print()
    print("Press 'q' to quit and print final values")
    print("Press 'r' to reset to defaults")
    
    while True:
        # Get trackbar values
        h_min = cv2.getTrackbarPos('H Min', 'HSV Tuner')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Tuner')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Tuner')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Tuner')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Tuner')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Tuner')
        
        # Create HSV range
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Show results
        scaled_imshow(img, scale=0.25, window_name='Original', waitkey=1)
        scaled_imshow(mask_clean, scale=0.25, window_name='Mask', waitkey=1)
        
        # Show current values on the tuner window
        tuner_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(tuner_img, f'H: {h_min}-{h_max}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(tuner_img, f'S: {s_min}-{s_max}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(tuner_img, f'V: {v_min}-{v_max}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(tuner_img, 'Press q to quit', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(tuner_img, 'Press r to reset', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('HSV Tuner', tuner_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\nFinal HSV Range:")
            print(f"lower_bound = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_bound = np.array([{h_max}, {s_max}, {v_max}])")
            break
        elif key == ord('r'):
            # Reset to defaults
            cv2.setTrackbarPos('H Min', 'HSV Tuner', 40)
            cv2.setTrackbarPos('H Max', 'HSV Tuner', 80)
            cv2.setTrackbarPos('S Min', 'HSV Tuner', 20)
            cv2.setTrackbarPos('S Max', 'HSV Tuner', 255)
            cv2.setTrackbarPos('V Min', 'HSV Tuner', 20)
            cv2.setTrackbarPos('V Max', 'HSV Tuner', 100)
    
    cv2.destroyAllWindows()

def analyze_color_pixels(image_path, show_analysis=True):
    """
    Analyze HSV values by clicking on pixels
    """
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hsv_values = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_pixel = hsv[y, x]
            bgr_pixel = img[y, x]
            hsv_values.append(hsv_pixel)
            
            print(f"Pixel ({x}, {y}):")
            print(f"  BGR: {bgr_pixel}")
            print(f"  HSV: H={hsv_pixel[0]}, S={hsv_pixel[1]}, V={hsv_pixel[2]}")
            print("-" * 30)
            
            # Draw a circle at clicked point
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Click Analysis', img)
    
    print("Click on different parts of the felt to analyze HSV values")
    print("Press any key when done")
    
    cv2.imshow('Click Analysis', img)
    cv2.setMouseCallback('Click Analysis', mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if hsv_values and show_analysis:
        hsv_array = np.array(hsv_values)
        print(f"\nAnalysis of {len(hsv_values)} clicked points:")
        print(f"H range: {hsv_array[:, 0].min()} - {hsv_array[:, 0].max()}")
        print(f"S range: {hsv_array[:, 1].min()} - {hsv_array[:, 1].max()}")
        print(f"V range: {hsv_array[:, 2].min()} - {hsv_array[:, 2].max()}")
        
        # Suggest HSV ranges with some tolerance
        h_min, h_max = max(0, hsv_array[:, 0].min() - 10), min(179, hsv_array[:, 0].max() + 10)
        s_min, s_max = max(0, hsv_array[:, 1].min() - 30), min(255, hsv_array[:, 1].max() + 30)
        v_min, v_max = max(0, hsv_array[:, 2].min() - 30), min(255, hsv_array[:, 2].max() + 30)
        
        print(f"\nSuggested HSV range (with tolerance):")
        print(f"lower_bound = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper_bound = np.array([{h_max}, {s_max}, {v_max}])")

# Usage examples
if __name__ == "__main__":
    image_path = '20250615_003443.jpg'  # Replace with your image path
    
    print("Choose tuning method:")
    print("1. Interactive HSV tuner with trackbars")
    print("2. Click analysis to sample HSV values")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        create_hsv_tuner(image_path)
    elif choice == "2":
        analyze_color_pixels(image_path)
    else:
        print("Invalid choice")