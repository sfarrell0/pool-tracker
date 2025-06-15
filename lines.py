import cv2
import numpy as np
import math

def calculate_line_length(line):
    """Calculate the length of a line segment"""
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_line_angle(line):
    """Calculate the angle of a line in degrees"""
    x1, y1, x2, y2 = line
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle


def _cartesian_to_polar(x1, y1, x2, y2):
    """
    Converts Cartesian line coordinates [x1, y1, x2, y2] to polar coordinates (rho, theta).
    Rho is the perpendicular distance from the origin to the line.
    Theta is the angle of the normal vector from the origin to the line, in radians [0, pi).
    Rho is always non-negative.

    Args:
        x1, y1, x2, y2 (float): Coordinates of the line's endpoints.

    Returns:
        tuple: (rho, theta) for the line.
    """
    # Handle degenerate line (a point)
    if x1 == x2 and y1 == y2:
        return 0.0, 0.0  # A point has infinite normals; (0,0) is a neutral representation

    # Line equation Ax + By + C = 0, where (A, B) is the normal vector
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1

    denominator = math.sqrt(A*A + B*B)
    
    # Calculate rho (perpendicular distance from origin to the line)
    rho = -C / denominator
    
    # Calculate theta (angle of the normal vector (A, B) with the positive x-axis)
    theta = math.atan2(B, A) # atan2(y, x) for vector (A, B)

    # Normalize rho and theta for consistent comparison:
    # 1. Ensure rho is non-negative. If rho is negative, negate it and adjust theta by pi.
    if rho < 0:
        rho = -rho
        theta += math.pi 
    
    # 2. Normalize theta to the range [0, pi).
    # First, ensure theta is within [0, 2pi)
    theta = theta % (2 * math.pi) 
    # If theta is in [pi, 2pi), map it to [0, pi) by subtracting pi
    if theta >= math.pi:
        theta -= math.pi
    
    # Handle floating point inaccuracies for angles very close to pi (e.g., pi - epsilon becomes 0)
    if abs(theta - math.pi) < 1e-9:
        theta = 0.0

    return rho, theta

def _merge_collinear_lines(group_lines):
    """
    Merges a group of lines (assumed to be collinear) into a single line
    by finding the farthest endpoints along their common direction.

    Args:
        group_lines (list of lists): A list of lines, each [x1, y1, x2, y2].

    Returns:
        list: The merged line [new_x1, new_y1, new_x2, new_y2].
    """
    if not group_lines:
        return []

    # Choose the first line as a reference for its direction vector
    ref_x1, ref_y1, ref_x2, ref_y2 = group_lines[0]

    dx = ref_x2 - ref_x1
    dy = ref_y2 - ref_y1
    
    length = math.sqrt(dx*dx + dy*dy)

    # Handle cases where the reference line is a point or very short.
    # If it's degenerate, try to find a non-degenerate line in the group.
    if length < 1e-6: # Using a small epsilon for floating point comparison
        for line in group_lines[1:]:
            temp_dx = line[2] - line[0]
            temp_dy = line[3] - line[1]
            if math.sqrt(temp_dx*temp_dx + temp_dy*temp_dy) >= 1e-6:
                ref_x1, ref_y1, ref_x2, ref_y2 = line
                dx = temp_dx
                dy = temp_dy
                length = math.sqrt(dx*dx + dy*dy)
                break
        if length < 1e-6: # If all lines in the group are degenerate, return the first one as is
             return [ref_x1, ref_y1, ref_x2, ref_y2] 

    # Calculate the unit direction vector of the reference line
    unit_vec_x = dx / length
    unit_vec_y = dy / length

    all_scalar_projections = []

    # Project all four endpoints of each line in the group onto the reference line
    # The scalar projection of point P (x_p, y_p) onto a line passing through R (ref_x1, ref_y1)
    # with unit direction vector U (unit_vec_x, unit_vec_y) is:
    # ((x_p - ref_x1) * unit_vec_x + (y_p - ref_y1) * unit_vec_y)
    for x1, y1, x2, y2 in group_lines:
        # Project endpoint 1
        vec_from_ref1_x = x1 - ref_x1
        vec_from_ref1_y = y1 - ref_y1
        all_scalar_projections.append(vec_from_ref1_x * unit_vec_x + vec_from_ref1_y * unit_vec_y)

        # Project endpoint 2
        vec_from_ref2_x = x2 - ref_x1
        vec_from_ref2_y = y2 - ref_y1
        all_scalar_projections.append(vec_from_ref2_x * unit_vec_x + vec_from_ref2_y * unit_vec_y)

    # Find the minimum and maximum scalar projections
    min_proj = min(all_scalar_projections)
    max_proj = max(all_scalar_projections)

    # Convert these minimum and maximum projected values back to Cartesian coordinates
    # New endpoint P_new = R + scalar_projection * U
    new_x1 = ref_x1 + min_proj * unit_vec_x
    new_y1 = ref_y1 + min_proj * unit_vec_y
    new_x2 = ref_x1 + max_proj * unit_vec_x
    new_y2 = ref_y1 + max_proj * unit_vec_y

    return [new_x1, new_y1, new_x2, new_y2]

def condense_lines(lines, rho_threshold=5.0, theta_threshold_deg=2.0):
    """
    Condenses a list of lines by combining those that are close in polar parameters
    (rho and theta) and extending their lengths to cover the farthest endpoints.

    Args:
        lines (list of lists): A list of lines, where each line is represented as
                               [x1, y1, x2, y2] (float coordinates).
        rho_threshold (float): The maximum allowed difference in 'rho' (distance from origin)
                                for two lines to be considered close. Default is 5.0 units.
        theta_threshold_deg (float): The maximum allowed difference in 'theta' (angle of normal)
                                     in degrees for two lines to be considered close.
                                     Default is 2.0 degrees.

    Returns:
        list of lists: A new list of condensed lines, each in [x1, y1, x2, y2] format.
    """
    if not lines:
        return []

    # Convert angular threshold from degrees to radians
    theta_threshold_rad = math.radians(theta_threshold_deg)

    # Prepare each line with its original coordinates, calculated polar coordinates,
    # and a flag to track if it has been processed (merged into a group).
    processed_lines_data = []
    for line in lines:
        x1, y1, x2, y2 = line
        rho, theta = _cartesian_to_polar(x1, y1, x2, y2)
        processed_lines_data.append({
            'original_coords': line,
            'polar_rho': rho,
            'polar_theta': theta,
            'processed': False  # Flag to indicate if this line has been grouped and merged
        })

    condensed_results = []

    # Iterate through all lines to find groups of close lines
    for i in range(len(processed_lines_data)):
        if processed_lines_data[i]['processed']:
            continue  # Skip lines that have already been merged

        current_line_data = processed_lines_data[i]
        current_rho = current_line_data['polar_rho']
        current_theta = current_line_data['polar_theta']

        # Start a new group with the current line
        group_to_merge = [current_line_data['original_coords']]
        processed_lines_data[i]['processed'] = True # Mark current line as processed

        # Compare the current line with all subsequent unprocessed lines
        for j in range(i + 1, len(processed_lines_data)):
            if processed_lines_data[j]['processed']:
                continue # Skip already processed lines

            compare_line_data = processed_lines_data[j]
            compare_rho = compare_line_data['polar_rho']
            compare_theta = compare_line_data['polar_theta']

            # Calculate differences in polar parameters
            rho_diff = abs(current_rho - compare_rho)
            theta_diff = abs(current_theta - compare_theta)
            
            # Check if lines are close enough in both rho and theta
            if rho_diff < rho_threshold and theta_diff < theta_threshold_rad:
                # If they are close, add to the current group and mark as processed
                group_to_merge.append(compare_line_data['original_coords'])
                processed_lines_data[j]['processed'] = True
        
        # After checking all other lines, merge all lines in the current group
        merged_line = _merge_collinear_lines(group_to_merge)
        condensed_results.append(merged_line)

    return condensed_results

def find_table_corners_from_edges(table_edges, extension_amount=50, img=None):
    """
    Extends a list of 2D line segments and finds all unique intersection points
    among the extended lines.

    Args:
        table_edges (list of lists): A 2D array where each inner list represents
                                      a line segment in the format [x1, y1, x2, y2].
        extension_amount (int/float): The amount in pixels to extend each side
                                      of the line segment. Defaults to 50.

    Returns:
        list of tuples: A list of (x, y) tuples, where each tuple is a unique
                        intersection point of the extended lines.
    """
    extended_lines = []
    # Using a set to automatically handle unique intersection points
    intersection_points = set()
    
    # --- Step 1: Extend each line segment ---
    for line in table_edges:
        x1, y1, x2, y2 = line
        
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate the length of the original line segment
        length = math.hypot(dx, dy)

        # Calculate the unit vector in the direction of the line
        unit_dx = dx / length
        unit_dy = dy / length
        
        # Extend the start point backward
        new_x1 = x1 - unit_dx * extension_amount
        new_y1 = y1 - unit_dy * extension_amount
        
        # Extend the end point forward
        new_x2 = x2 + unit_dx * extension_amount
        new_y2 = y2 + unit_dy * extension_amount
        
        extended_lines.append([new_x1, new_y1, new_x2, new_y2])
        
    # --- Step 2: Find all intersection points among the extended lines ---
    num_extended_lines = len(extended_lines)
    
    # Iterate through all unique pairs of extended lines
    for i in range(num_extended_lines):
        for j in range(i + 1, num_extended_lines): # Start from i+1 to avoid duplicate pairs and self-intersection
            line1_coords = extended_lines[i]
            line2_coords = extended_lines[j]
            
            x1, y1, x2, y2 = line1_coords
            x3, y3, x4, y4 = line2_coords
            
            # Calculate the denominator for the intersection formula
            # This value determines if lines are parallel or intersect
            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            # Use a small epsilon to account for floating-point inaccuracies when checking for parallelism
            epsilon = 1e-9 
            if abs(denominator) < epsilon:
                # Lines are parallel or collinear; they do not have a single unique intersection point
                continue 
            
            # Calculate the numerators for the intersection point coordinates
            numerator_x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
            numerator_y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
            
            # Calculate the intersection point
            intersection_x = numerator_x / denominator
            intersection_y = numerator_y / denominator

            match = False
            for point in intersection_points:
                if abs(intersection_x - point[0]) < 10 and abs(intersection_y - point[1]) < 10:
                    match = True
            if img is not None:
                if not (intersection_x > 0 and intersection_x < img.shape[1] and intersection_y > 0 and intersection_y < img.shape[0]):
                    match = True
            if not match:
                # Add the intersection point (as a tuple for hashability) to the set
                intersection_points.add((int(intersection_x), int(intersection_y)))
            
    return list(intersection_points) # Convert the set back to a list before returning

