import cv2
import numpy as np
import math
from utils import scaled_imshow
from lines import calculate_line_length, calculate_line_angle, condense_lines

def detect_black_table(image):
    """Detect black pool table using threshold 140 and finding largest black contour"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    
    # Create mask for dark areas (below threshold 140)
    black_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    black_mask[v_channel < 140] = 255
    
    # scaled_imshow(black_mask, 0.25, "Black Areas (< 140)")
    return black_mask

def find_largest_black_contour(black_mask, original_image):
    """Find the largest black contour and clean it up"""
    

    EROSION_ITERAITIONS = 2
    kernel_erode = np.ones((5,5), np.uint8)
    eroded = cv2.erode(black_mask, kernel_erode, iterations=EROSION_ITERAITIONS)
    # cv2.imshow("Eroded", eroded)

    cleaned = eroded.copy()
    
    # Find all contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    
    
    # Create mask with only the largest contour
    table_mask = np.zeros_like(cleaned)
    cv2.drawContours(table_mask, [largest_contour], -1, 255, -1)
    #Undo erosion to restore original size
    table_mask = cv2.dilate(table_mask, kernel_erode, iterations=EROSION_ITERAITIONS)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h
    
    # Draw bounding rectangle on original image
    result_image = original_image.copy()
    cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(result_image, f"Table AR: {aspect_ratio:.2f}", 
               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # scaled_imshow(result_image, 0.25, "Detected Table (Bounding Box)")
    
    # Also fit a rotated rectangle for better accuracy
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    rotated_result = original_image.copy()
    cv2.drawContours(rotated_result, [box], 0, (255, 0, 0), 3)
    
    # Calculate rotated aspect ratio
    width, height = rect[1]
    if width > height:
        rot_aspect_ratio = width / height
    else:
        rot_aspect_ratio = height / width
        
    cv2.putText(rotated_result, f"Rotated AR: {rot_aspect_ratio:.2f}", 
               (int(rect[0][0]), int(rect[0][1])-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # scaled_imshow(rotated_result, 0.25, "Rotated Rectangle Fit")
    
    return table_mask, (x, y, w, h), rect

def detect_table_edges_from_contour(table_mask, original_image, min_line_length=150):
    """Use table contour to detect table edges and filter by length"""
    
    # Find edges of the table region
    edges = cv2.Canny(table_mask, 50, 150)
    # scaled_imshow(edges, 1, "Table Edges")
    
    # Dilate edges to connect nearby segments
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    # scaled_imshow(dilated_edges, 1, "Dilated Edges")
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=200, 
                           minLineLength=100, maxLineGap=100)
    
    line_image = original_image.copy()
    filtered_lines = []
    
    if lines is not None:

        # Filter lines by length
        for line in lines:
            line = line[0]  # Extract the line coordinates
            length = calculate_line_length(line)
            if length >= min_line_length:
                filtered_lines.append(line)

        # Condense lines by merging close and similar lines
        #lines = [line[0] for line in filtered_lines]
        condensed_lines = condense_lines(filtered_lines)
        condensed_lines = [list(map(int, map(round, line))) for line in condensed_lines]
                
        
        # Draw filtered lines in bright color
        for line in condensed_lines:
            x1, y1, x2, y2 = line
            length = calculate_line_length(line)
            angle = calculate_line_angle(line)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Add length annotation
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(line_image, f"{length:.0f}", (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    scaled_imshow(line_image, 1, "Filtered Table Edge Lines")


    return condensed_lines


def main():
    # Load the image
    # image = cv2.imread('20250615_003443.jpg')
    # image = cv2.imread('20250615_003453.jpg')
    # image = cv2.imread('20250615_003515.jpg')
    # image = cv2.imread('20250615_003522.jpg')
    image = cv2.imread('20250615_003529.jpg')
    if image is None:
        print("Error: Could not load image. Check file path.")
        return
    
    
    scaled_imshow(image, 0.1, "Original Image")

    black_mask = detect_black_table(image)

    table_mask, bbox, rotated_rect = find_largest_black_contour(black_mask, image)
    
    if table_mask is not None:
        lines = detect_table_edges_from_contour(table_mask, image, min_line_length=150)
        print(lines)
        
        if bbox:
            x, y, w, h = bbox
            print(f"\nOriginal detection:")
            print(f"  Position: ({x}, {y})")
            print(f"  Size: {w} x {h}")
            print(f"  Aspect ratio: {w/h:.2f}")
            
            # Extract table region for comparison
            table_region = image[y:y+h, x:x+w]
            # scaled_imshow(table_region, 0.5, "Original Extracted Table Region")
    else:
        print("Could not detect table!")
    
    print("\nDetection complete. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()