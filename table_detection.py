import cv2
import numpy as np
import math
from utils import scaled_imshow
from lines import calculate_line_length, calculate_line_angle, condense_lines

def detect_black_table(image):
    """Detect black pool table using threshold 140 and finding largest black contour"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]

    # Create mask for dark areas (below threshold 105)
    black_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    black_mask[v_channel < 105] = 255
    
    # scaled_imshow(black_mask, 0.25, "Black Areas (< 140)")
    return black_mask

def find_largest_black_contour(black_mask, original_image):
    """Find the largest black contour and clean it up"""
    

    EROSION_ITERAITIONS = 6
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
    
    return table_mask

def detect_table_edges_from_contour(table_mask, original_image, min_line_length=150):
    """Use table contour to detect table edges and filter by length"""
    
    # Find edges of the table region
    edges = cv2.Canny(table_mask, 50, 150)
    #scaled_imshow(edges, 1, "Table Edges")
    
    # Dilate edges to connect nearby segments
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=0)

    # Find contours on the dilated edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly_img = original_image.copy()
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(poly_img, [approx], -1, (0, 255, 255), 2)
    cv2.imshow("Polygon Approximation", poly_img)
    #scaled_imshow(dilated_edges, 1, "Dilated Edges")

    lines_hough = cv2.HoughLines(dilated_edges, 1, np.pi/180, threshold=170)
    test_img = original_image.copy()
    lines = []
    if lines_hough is not None:
        for line in lines_hough:
            print(f"Line: {line}")
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 5000 * (-b))
            y1 = int(y0 + 5000 * (a))
            x2 = int(x0 - 5000 * (-b))
            y2 = int(y0 - 5000 * (a))
            lines.append([[x1, y1, x2, y2]])
            cv2.line(test_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Detected Lines', test_img)
    
    # Find lines using Hough transform
    # lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=200, 
    #                        minLineLength=100, maxLineGap=100)
    
    line_image = original_image.copy()
    filtered_lines = []
    condensed_lines = []
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
    
    #scaled_imshow(line_image, 1, "Filtered Table Edge Lines")


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