import time
import math
import cv2
import torch
import os
from djitellopy import Tello
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



# --------------------- DroneState Class --------------------- #
class DroneState:
    def __init__(self):
        self.x = 0  # Forward/Backward position in cm
        self.y = 0  # Left/Right position in cm
        self.z = 50  # Altitude in cm (assuming starting at 50 cm)
        self.yaw = 0  # Orientation in degrees

    def update_position(self, distance, direction):
        """
        Update the drone's position based on movement.

        Args:
            distance (int): Distance moved in cm.
            direction (str): Direction moved ('forward', 'backward', 'left', 'right').
        """
        rad = math.radians(self.yaw)
        if direction == 'forward':
            self.x += distance * math.cos(rad)
            self.y += distance * math.sin(rad)
        elif direction == 'backward':
            self.x -= distance * math.cos(rad)
            self.y -= distance * math.sin(rad)
        elif direction == 'left':
            # Moving left relative to current orientation
            self.x += distance * math.sin(rad)
            self.y -= distance * math.cos(rad)
        elif direction == 'right':
            # Moving right relative to current orientation
            self.x -= distance * math.sin(rad)
            self.y += distance * math.cos(rad)

    def update_yaw(self, angle):
        """
        Update the drone's yaw.

        Args:
            angle (int): Angle to rotate. Positive for CCW, negative for CW.
        """
        self.yaw = (self.yaw + angle) % 360

# --------------------- Movement Functions --------------------- #
def move_drone(tello, drone_state, direction, distance): 
    """
    Simulate moving the drone in a specified direction by a specified distance.

    Args:
        drone_state (DroneState): The current state of the drone.
        direction (str): Direction to move ('forward', 'backward', 'left', 'right', 'up', 'down').
        distance (int): Distance to move in cm.
    """
    
    if direction == 'forward':
        tello.move_forward(distance)
    elif direction == 'backward':
        tello.move_back(distance)
    elif direction == 'left':
        tello.move_left(distance)
    elif direction == 'right':
        tello.move_right(distance)
    elif direction == 'up':
        tello.move_up(distance)
    elif direction == 'down':
        tello.move_down(distance)
    
    drone_state.update_position(distance, direction)
    print(f"Drone moved {direction} by {distance} cm. New position: (x={drone_state.x:.2f}, y={drone_state.y:.2f}, z={drone_state.z:.2f})")
    time.sleep(distance/70)  # Simulate movement time

def rotate_drone(tello, drone_state, angle):
    """
    Simulate rotating the drone by a specified angle.

    Args:
        drone_state (DroneState): The current state of the drone.
        angle (int): Angle to rotate. Positive for CCW, negative for CW.
    """

    
    if angle > 0:
        tello.rotate_counter_clockwise(angle)
    else:
        #tello.rotate_clockwise(-angle)
        tello.rotate_clockwise(-angle)
    
    drone_state.update_yaw(angle)
    direction = "CCW" if angle > 0 else "CW"
    print(f"Drone rotated {direction} by {abs(angle)} degrees. New yaw: {drone_state.yaw} degrees")
    # Estimate rotation time based on angle (approx. 1 sec per 90 degrees)
    time.sleep(2)  # Simulate rotation time

# --------------------- Movement Sequence --------------------- #
def define_square_movement(side_length, rotation_angle, capture_rotation):
    """
    Define a square movement pattern with image capture at each corner.

    Args:
        side_length (int): Length of each side in cm.
        rotation_angle (int): Angle to rotate at each corner.
        capture_rotation (int): Angle to rotate for capturing images at each corner.
    
    Returns:
        List of tuples representing movements: (action, value)
    """
    movements = []
    for _ in range(4):
        movements.append(('forward', side_length))
        movements.append(('rotate_for_capture', rotation_angle))
        movements.append(('return_rotate', capture_rotation))
    return movements

# --------------------- Capture Image Function --------------------- #
def capture_image(tello, drone_state, waypoint_number):
    """
    Capture an image from the drone's camera and return it along with the current position and orientation.

    Args:
        tello (Tello): The Tello drone instance.
        drone_state (DroneState): The current state of the drone.
        waypoint_number (int): The current waypoint number.

    Returns:
        Tuple containing the frame, position, and orientation.
    """
    try:
        frame = tello.get_frame_read().frame
        position = {'x': drone_state.x, 'y': drone_state.y, 'z': drone_state.z}
        orientation = drone_state.yaw
        return frame, position, orientation
    except Exception as e:
        print(f"Error capturing image at waypoint {waypoint_number}: {e}")
        return None, None, None

# --------------------- Process Frame Function --------------------- #
def process_frame(frame, yolo_model, midas, transform, device):
    """
    Process an image frame to obtain annotated frame, detection results, and depth map.

    Args:
        frame (np.ndarray): Input image frame.
        yolo_model: YOLO model instance.
        midas: MiDaS model instance.
        transform: MiDaS transform function.
        device: Torch device.

    Returns:
        Annotated frame (np.ndarray), YOLO detection results, depth map (np.ndarray)
    """
    # Perform object detection
    results = yolo_model(frame)[0]

    # Perform depth estimation
    input_batch = transform(frame).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()

    # Optional: Apply median filter to reduce noise
    #depth_map = cv2.medianBlur(depth_map, 5)


    # Normalize depth map for visualization (optional)
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map_visual = (depth_map - depth_min) / (depth_max - depth_min)
    depth_map_visual = (depth_map_visual * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_MAGMA)
    depth_frame = depth_colored

    # Annotate frame with detections
    annotated_frame = frame.copy()
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
        cls = int(result.cls[0].item())
        confidence = float(result.conf[0].item())
        class_label = yolo_model.names[cls]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{class_label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_frame, results, depth_map, depth_frame

# --------------------- Map Objects Function --------------------- #
def map_objects(results, depth_map, camera_intrinsics, position, orientation, yolo_model, sampling_rate=10):
    """
    Map detected objects' bounding boxes to world coordinates using the corner points and their depths.
    
    Args:
        results: Detection results from YOLO.
        depth_map: Depth information from MiDaS.
        camera_intrinsics (dict): Camera intrinsic parameters.
        position (dict): Current position of the drone.
        orientation (float): Current orientation (yaw) of the drone in degrees.
        yolo_model: YOLO model instance.
        sampling_rate (int): Step size for sampling pixels within bounding boxes.
    
    Returns:
        List of mapped objects with world bounding boxes in world coordinates and size information.
    """
    mapped_objects = []
    image_height, image_width = depth_map.shape

    for result in results.boxes:
        # Extract bounding box and class information
        bbox = result.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
        cls = int(result.cls[0].item())
        confidence = float(result.conf[0].item())
        class_label = yolo_model.names[cls]

        # Ensure bounding box is within image bounds
        x1, y1, x2, y2 = bbox
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, image_width - 1)
        y2 = min(y2, image_height - 1)

        # Define the four corner points of the bounding box
        corners = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x1, y2),  # Bottom-left
            (x2, y2)   # Bottom-right
        ]

        valid_corners = []

        for (x, y) in corners:
            # Retrieve depth at corner
            depth = depth_map[y, x]
            if depth > 0:
                valid_corners.append((x, y, depth))
            else:
                # If corner depth is invalid, try sampling nearby pixels
                window_size = 5  # Search within a 5x5 window
                found = False
                for dx in range(-window_size, window_size + 1):
                    for dy in range(-window_size, window_size + 1):
                        nx = np.clip(x + dx, 0, image_width - 1)
                        ny = np.clip(y + dy, 0, image_height - 1)
                        sampled_depth = depth_map[ny, nx]
                        if sampled_depth > 0:
                            valid_corners.append((nx, ny, sampled_depth))
                            found = True
                            break
                    if found:
                        break
            # If still not found, skip this corner
        if len(valid_corners) < 4:
            print(f"Insufficient valid depth data for object '{class_label}'. Skipping.")
            continue

        # Convert each valid corner to world coordinates
        world_corners = []
        size_x_cm_list = []
        size_y_cm_list = []
        for (x, y, depth) in valid_corners:
            fx = camera_intrinsics['fx']
            fy = camera_intrinsics['fy']
            cx = camera_intrinsics['cx']
            cy = camera_intrinsics['cy']

            # Convert pixel coordinates to camera coordinates
            X_cam = (x - cx) * depth / fx
            Y_cam = (cy - y) * depth / fy
            #Z_cam = depth  
            
            
            # Rotate based on drone orientation (yaw)
            yaw_rad = math.radians(orientation)
            X_world = X_cam * math.cos(yaw_rad) - Y_cam * math.sin(yaw_rad)
            Y_world = X_cam * math.sin(yaw_rad) + Y_cam * math.cos(yaw_rad)
            # Z_world = Z_cam + position['z']  # Not used for 2D mapping

            world_corners.append((X_world, Y_world))

        # Determine the bounding box in world coordinates
        xs, ys = zip(*world_corners)
        min_x_world = min(xs)
        max_x_world = max(xs)
        min_y_world = min(ys)
        max_y_world = max(ys)

        world_bbox = [
            min_x_world,
            min_y_world,
            max_x_world,
            max_y_world
        ]

        # Compute center and size
        center_x = (min_x_world + max_x_world) / 2
        center_y = (min_y_world + max_y_world) / 2
        width_cm = max_x_world - min_x_world
        height_cm = max_y_world - min_y_world
        size_cm = (width_cm + height_cm) / 2  # Average size for dot scaling

        # Append the mapped object with its details
        mapped_objects.append({
            'type': cls,
            'class_label': class_label,
            'confidence': confidence,
            'world_bbox': world_bbox,            # [min_x, min_y, max_x, max_y]
            'center_x': center_x,                # Center X coordinate
            'center_y': center_y,                # Center Y coordinate
            'size_cm': size_cm                     # Average bounding box size in cm
        })

        print(f"Mapped object: {class_label} | Confidence: {confidence:.2f} | Center: ({center_x:.2f}, {center_y:.2f}) cm | Size: {size_cm:.2f} cm")

    return mapped_objects

# --------------------- Deduplicate Detections Function --------------------- #
def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA: List or array with format [min_x, min_y, max_x, max_y]
        boxB: List or array with format [min_x, min_y, max_x, max_y]

    Returns:
        IoU value (float)
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of the bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

'''def deduplicate_objects(mapped_objects, iou_threshold=0.3):
    """
    Deduplicate objects based on IoU overlap for the same class.

    Args:
        mapped_objects (list): List of mapped objects with world bounding boxes.
        iou_threshold (float): Threshold above which objects are considered duplicates.

    Returns:
        List of deduplicated objects.
    """
    deduplicated = []
    for obj in mapped_objects:
        duplicate_found = False
        for dedup_obj in deduplicated:
            if obj['class_label'] != dedup_obj['class_label']:
                continue  # Only compare objects of the same class

            iou = calculate_iou(obj['world_bbox'], dedup_obj['world_bbox'])
            if iou >= iou_threshold:
                # If duplicate, keep the one with higher confidence
                if obj['confidence'] > dedup_obj['confidence']:
                    deduplicated.remove(dedup_obj)
                    deduplicated.append(obj)
                duplicate_found = True
                break
        if not duplicate_found:
            deduplicated.append(obj)
    return deduplicated
'''

def deduplicate_with_dbscan(mapped_objects, eps=10, min_samples=1):
    """
    Deduplicate objects using DBSCAN clustering based on their spatial coordinates.
    Retains the object with the highest confidence in each cluster.
    
    Args:
        mapped_objects (list): List of mapped objects with 'center_x', 'center_y', and 'confidence'.
        eps (float): The maximum distance between two samples for them to be considered in the same neighborhood (in cm).
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
        List of deduplicated objects.
    """
    if not mapped_objects:
        return []
    
    # Extract coordinates
    coords = np.array([[obj['center_x'], obj['center_y']] for obj in mapped_objects])
    
    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps/10, min_samples=min_samples).fit(coords)
    labels = db.labels_
    
    # Number of clusters, ignoring noise if present
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label if present
    
    deduplicated = []
    
    for label in unique_labels:
        # Indices of objects in the current cluster
        indices = np.where(labels == label)[0]
        cluster_objects = [mapped_objects[idx] for idx in indices]
        
        # Select the object with the highest confidence
        best_obj = max(cluster_objects, key=lambda x: x['confidence'])
        deduplicated.append(best_obj)
    
    # Handle noise points (if any) by treating them as unique objects
    noise_indices = np.where(labels == -1)[0]
    noise_objects = [mapped_objects[idx] for idx in noise_indices]
    deduplicated.extend(noise_objects)
    
    print(f"Total objects before DBSCAN deduplication: {len(mapped_objects)}")
    print(f"Total objects after DBSCAN deduplication: {len(deduplicated)}")
    
    return deduplicated

# --------------------- Create Separate 2D Map Function --------------------- #
def create_separate_2d_maps(object_map, output_path="processed_image/separate_2d_maps.png"):
    """
    Create separate 2D maps for each image, plotting objects as dots with sizes proportional to their original bounding box sizes.
    
    Args:
        object_map (dict): Dictionary with image identifiers as keys and lists of mapped objects as values.
        output_path (str): Path to save the generated separate 2D maps.
    
    Returns:
        None
    """
    num_images = len(object_map)
    if num_images == 0:
        print("No objects to plot in separate maps.")
        return

    # Determine grid size (e.g., 2x2 for 4 images)
    grid_size = math.ceil(math.sqrt(num_images))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))  # Increased figure size for better visibility
    axs = axs.flatten()

    color_cycle = plt.get_cmap('tab20').colors
    class_colors = {}
    current_color = 0
    

    for idx, (image_id, objects) in enumerate(object_map.items()):
        ax = axs[idx]
        if not objects:
            ax.set_title(f"Image: {image_id} (No Objects)")
            ax.set_xlabel("X (relative distance)")
            ax.set_ylabel("Y (relative distance)")
            ax.grid(True)
            ax.axis('equal')  # Equal scaling for both axes
            continue

        # Determine axis limits for this subplot
        all_x = [obj['center_x'] for obj in objects]
        all_y = [obj['center_y'] for obj in objects]
        
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        
        padding_x = (max_x - min_x) * 2 if (max_x - min_x) > 0 else 10
        padding_y = (max_y - min_y) * 2 if (max_y - min_y) > 0 else 10
        
        axis_min_x = min_x - padding_x
        axis_max_x = max_x + padding_x
        axis_min_y = min_y - padding_y
        axis_max_y = max_y + padding_y
        
        
        for obj in objects:
            class_label = obj['class_label']
            if class_label not in class_colors:
                class_colors[class_label] = color_cycle[current_color % len(color_cycle)]
                current_color += 1
            color = class_colors[class_label]

            center_x = obj['center_x']
            center_y = obj['center_y']
            size_cm = obj['size_cm']
            size_points = (size_cm ** 2) * 5  # Adjust scaling factor as needed

            # Plot a scatter point
            ax.scatter(center_x, center_y, s=size_points, color=color, alpha=0.6, label=class_label)

            # Optionally, annotate the dot with class label and confidence
            ax.text(center_x, center_y, f"{class_label}\n{obj['confidence']:.2f}", fontsize=5,
                    color='black', ha='center', va='center')

        ax.set_title(f"Image: {image_id}")
        ax.set_xlabel("X (relative distance)")
        ax.set_ylabel("Y (relative distance)")
        ax.grid(True)
        ax.axis('equal')  # Equal scaling for both axes
        ax.set_xlim(axis_min_x, axis_max_x)
        ax.set_ylim(axis_min_y, axis_max_y)

    # Hide any unused subplots
    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])

    # Increase spacing between subplots to prevent them from being too close
    plt.subplots_adjust(wspace=1, hspace=1)  # Adjust these values as needed (default is 0.2)

    '''# Create a single legend for all classes to avoid duplicate entries
    handles, labels = axs[0].get_legend_handles_labels()
    if handles and labels:
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper right')
    '''
    # Adjust layout to make space for the legend
    plt.tight_layout()  # Leave space on the right for the legend


    # Save the figure
    plt.savefig(output_path)
    plt.show()
    print(f"Separate 2D maps saved to {output_path}.")

# --------------------- Create Merged 2D Map Function --------------------- #

def create_merged_map(object_map, output_path="processed_image/merged_2d_map.png", eps=10, min_samples=1):
    """
    Create a merged 2D map by aligning objects based on unique reference objects,
    deduplicating them, and plotting as dots with sizes proportional to their original bounding box sizes.
    
    Args:
        object_map (dict): Dictionary with image identifiers as keys and lists of mapped objects as values.
        output_path (str): Path to save the generated merged 2D map.
    
    Returns:
        None
    """
    # Step 1: Identify Unique Reference Classes
    # A unique reference class appears exactly once in each image it exists
    class_occurrences = defaultdict(int)
    class_image_counts = defaultdict(set)
    
    for image_id, objects in object_map.items():
        class_counts = defaultdict(int)
        for obj in objects:
            class_counts[obj['class_label']] += 1
        for cls, count in class_counts.items():
            if count == 1:
                class_occurrences[cls] += 1
                class_image_counts[cls].add(image_id)
    
    # Step 2: Identify Potential Reference Classes
    # A class is a potential reference if it appears only once per image and exists in at least two images
    potential_refs = [cls for cls, count in class_occurrences.items() if count >= 2]
    
    # Step 3: Determine the Best Reference Class
    # Prefer classes that appear in the most images
    if not potential_refs:
        print("No unique reference objects found across any images. Merged map will not be created.")
        return
    
    # Sort potential_refs by the number of images they appear in, descending
    potential_refs.sort(key=lambda cls: len(class_image_counts[cls]), reverse=True)
    reference_class = potential_refs[0]  # Choose the class that appears in the most images
    
    images_with_ref = class_image_counts[reference_class]
    num_refs = len(images_with_ref)
    
    if num_refs < 2:
        print(f"Only {num_refs} images have the reference object '{reference_class}'. Merged map will not be created.")
        return
    
    print(f"Using '{reference_class}' as the reference class for merging across {num_refs} images.")
    
    # Step 4: Collect Reference Object Coordinates
    reference_coords = {}
    for image_id in images_with_ref:
        for obj in object_map[image_id]:
            if obj['class_label'] == reference_class:
                center_x = obj['center_x']
                center_y = obj['center_y']
                reference_coords[image_id] = (center_x, center_y)
                break
    
    # Step 5: Choose a Base Reference
    base_image_id = next(iter(images_with_ref))
    base_ref_x, base_ref_y = reference_coords[base_image_id]
    
    # Step 6: Align Objects Based on Reference
    merged_objects = []
    for image_id, objects in object_map.items():
        if image_id in reference_coords:
            ref_x, ref_y = reference_coords[image_id]
            dx = base_ref_x - ref_x
            dy = base_ref_y - ref_y
            for obj in objects:
                # Shift center points based on reference
                obj_shifted = obj.copy()
                obj_shifted['center_x'] += dx
                obj_shifted['center_y'] += dy
                merged_objects.append(obj_shifted)
        else:
            # Skip images without the reference object
            print(f"Image '{image_id}' does not have the reference object '{reference_class}'. Skipping.")
            continue
    
    if not merged_objects:
        print("No objects to plot after alignment. Merged map will not be created.")
        return
    
    # Step 7: Deduplicate Merged Objects
    deduplicated_objects = deduplicate_with_dbscan(merged_objects, eps=eps, min_samples=min_samples)
    print(f"Total objects before deduplication: {len(merged_objects)}")
    print(f"Total objects after deduplication: {len(deduplicated_objects)}")
    
    if not deduplicated_objects:
        print("No objects to plot after deduplication. Merged map will not be created.")
        return
    
    # Step 8: Determine Global Axis Limits
    all_x = [obj['center_x'] for obj in deduplicated_objects]
    all_y = [obj['center_y'] for obj in deduplicated_objects]
    
    if not all_x or not all_y:
        print("No valid object coordinates found for plotting.")
        return
    
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    
    
    # Add padding to the axis limits
    padding_x = (max_x - min_x) * 2 if (max_x - min_x) > 0 else 10
    padding_y = (max_y - min_y) * 2 if (max_y - min_y) > 0 else 10
    
    axis_min_x = min_x - padding_x
    axis_max_x = max_x + padding_x
    axis_min_y = min_y - padding_y
    axis_max_y = max_y + padding_y
    


    
    # Step 9: Plot the Deduplicated Objects as Dots
    fig, ax = plt.subplots(figsize=(12, 12))  # Increased figure size for better visibility
    
    # Define a color map for different classes
    class_colors = {}
    color_cycle = plt.get_cmap('tab20').colors
    current_color = 0
    
    # Define a scaling factor for dot sizes (adjust as needed)
    scaling_factor = 5  # Example: 5 cm corresponds to size 25
    
    for obj in deduplicated_objects:
        class_label = obj['class_label']
        if class_label not in class_colors:
            class_colors[class_label] = color_cycle[current_color % len(color_cycle)]
            current_color += 1
        color = class_colors[class_label]
    
        center_x = obj['center_x']
        center_y = obj['center_y']
        size_cm = obj['size_cm']
        size_points = (size_cm ** 2) * scaling_factor  # Scale area based on size_cm
    
        # Plot a scatter point
        ax.scatter(center_x, center_y, s=size_points, color=color, alpha=0.6, label=class_label)
    
        # Optionally, annotate the dot with class label and confidence
        ax.text(center_x, center_y, f"{class_label}\n{obj['confidence']:.2f}", fontsize=5,
                color='black', ha='center', va='center')
    
    ax.set_title("Merged 2D Object Map with Alignment and Deduplication (Dots Represent Objects)")
    ax.set_xlabel("X (relative distance)")
    ax.set_ylabel("Y (relative distance)")
    ax.grid(True)
    ax.axis('equal')  # Equal scaling for both axes
    ax.set_xlim(axis_min_x, axis_max_x)
    ax.set_ylim(axis_min_y, axis_max_y)
    
    '''# Create legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')'''
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(output_path)
    plt.show()
    print(f"Merged 2D map saved to {output_path}.")



# --------------------- Main Function --------------------- #
def main():
    # --------------------- Initialization Inside main() --------------------- #

    # Initialize Object Map
    object_map = {}

    # Initialize Drone State
    drone_state = DroneState()

    image_folder = "scan_image"
    os.makedirs(image_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Initialize Drone
    tello = Tello()
    try:
        tello.connect()
        battery_level = tello.get_battery()
        print(f"Connected to Tello Drone. Battery Level: {battery_level}%")

        if battery_level < 20:
            print("Battery too low. Please charge the drone before starting.")
            return

        # Enable SDK Live Video
        tello.streamon()

        # Load YOLO Model
        yolo_model = YOLO('weights/yolo11n.pt', task='detect')

        # Load MiDaS Model
        model_type = "DPT_Large"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, verbose=False)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        # Load MiDaS Transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        # Define Square Movement Parameters
        side_length = 450  # cm
        rotation_angle = 135  # degrees to face inward
        return_rotation = -45  # degrees to face straight for flight
        movements = define_square_movement(side_length, rotation_angle, return_rotation)

        # Define Camera Intrinsics (These values should be calibrated for your drone's camera)
        camera_intrinsics = {
            'fx': 600,  # Example value, adjust based on actual calibration
            'fy': 600,
            'cx': 320,  # Assuming image width of 640 pixels
            'cy': 240   # Assuming image height of 480 pixels
        }

        # --------------------- Main Flight and Processing Loop --------------------- #

        # Takeoff
        print("Initiating takeoff...")

        tello.takeoff()
        time.sleep(2)  # Wait for drone to stabilize in the air
        tello.move_up(100)
        time.sleep(2)
        print("Takeoff complete.")
        print("Starting flight and data capture...")

        ##First Capture to initiate
        frame = tello.get_frame_read().frame

        for idx, movement in enumerate(movements, 1):
            action, value = movement
            print(f"Executing movement {idx}: {action} {value}")

            if action in ['forward', 'backward', 'left', 'right', 'up', 'down']:
                move_drone(tello, drone_state, action, value)

                
            elif action == 'return_rotate':
                rotate_drone(tello, drone_state, value)

                #Inference here for correct yaw
                position = {'x': drone_state.x, 'y': drone_state.y, 'z': drone_state.z}
                orientation = drone_state.yaw

                print(f"To map position: {position} with orientation: {orientation} ")
                
                if frame is not None:
                    print(f"Processing Frame at waypoint {idx}...")
                    position = {'x': drone_state.x, 'y': drone_state.y, 'z': drone_state.z}
                    orientation = drone_state.yaw
                    print(f"To map position: {position} with orientation: {orientation} degrees")
                        
                    print(f"Processing Frame at waypoint {idx}...")
                    annotated_frame, results, depth_map, depth_frame = process_frame(frame, yolo_model, midas, transform, device)
                    mapped_objects = map_objects(results, depth_map, camera_intrinsics, position, orientation, yolo_model)
                    
                    image_pointer = f"waypoint_{idx}_capture_raw.jpg"
                    object_map[image_pointer] = mapped_objects


                    # Optional: Display Annotated Frame for Verification
                    cv2.imshow(f"Waypoint {idx} - Annotated Frame", annotated_frame)
                    
                    # Save the annotated image
                    image_filename = os.path.join(image_folder, f"waypoint_{idx}_capture_raw.jpg")
                    cv2.imwrite(image_filename, frame)
                    
                    output_image_path = os.path.join(image_folder, f"waypoint_{idx}_capture_annotated.jpg")
                    cv2.imwrite(output_image_path, annotated_frame)
                    print(f"Annotated image saved to {output_image_path}.")
                    
                    output_image_path = os.path.join(image_folder, f"waypoint_{idx}_capture_depth.jpg")
                    cv2.imwrite(output_image_path, depth_frame)
                    print(f"depth image saved to {output_image_path}.")

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Flight interrupted by user.")
                        break
                else:
                    print(f"Skipping processing at waypoint {idx} due to capture error.")

                
            elif action == 'rotate_for_capture':
                # 1. Rotate inward by capture_rotation degrees
                rotate_drone(tello, drone_state, value)
                
                # 2. Capture and process image
                print(f"Capturing image at waypoint {idx} after rotating inward by {value} degrees.")
                frame, position, orientation = capture_image(tello, drone_state, idx)
                

                """
                # 3. Rotate back by -capture_rotation degrees to original orientation
                rotate_drone(tello, drone_state, -value)
                
                # 4. Rotate to turn to next side by rotation_angle degrees
                rotate_drone(tello, drone_state, rotation_angle)
                """
                
            else:
                print(f"Unknown action: {action}")
                continue

        # After processing all waypoints, proceed to plotting
        
        print("\nGenerating separate 2D object maps...")
        create_separate_2d_maps(object_map)
        print("Separate 2D object maps generation complete.")

        print("\nGenerating merged 2D object map...")
        create_merged_map(object_map)
        print("Merged 2D object map generation complete.")

        '''
        # Display all OpenCV windows
        print("\nDisplaying annotated frames. Press 'q' in any OpenCV window to exit.")
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        '''
        
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Land the drone
        print("Initiating landing...")
        tello.land()
        time.sleep(2)  # Wait for drone to land
        print("Landing complete.")

        # Cleanup
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()
        print("Exiting...")

if __name__ == "__main__":
    main() 