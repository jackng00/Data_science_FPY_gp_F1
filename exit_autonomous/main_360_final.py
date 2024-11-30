import logging
import csv
from json import load
from djitellopy import Tello
from time import sleep
from ExitFinding import *
from dronemission import *
from threading import Thread, Event
import cv2
import numpy as np
import os
from orb_process import Frame, denormalize, match_frames, add_ones
from orb_process import Map, Point
from queue import Queue, Empty
from numpy import array, asarray, max, vstack
from math import sqrt, tan, degrees
from open3d.cpu.pybind.io import read_point_cloud
from pandas import DataFrame
from utils import *




# Drone movement functions
def move(drone):
    """
    Moves the drone up and down for focus adjustment.
    """
    try:
        drone.move_up(20)
        sleep(1.5)  # Allow time for movement
        drone.move_down(20)
        sleep(1.5)
    except Exception as e:
        logging.error(f"Error during drone movement: {e}")

# Triangulation function remains unchanged
def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[-1]  # Corrected index to -1 for the last row

    return ret

# Process frame function with added parameters
def process_frame(img, mapp, K, W, H):
    """
    Processes a single frame to extract and triangulate 3D points.

    Args:
        img (np.ndarray): The image frame.
        mapp (Map): The map object to store 3D points.
        K (np.ndarray): Camera intrinsic matrix.
        W (int): Image width.
        H (int): Image height.
    
    Returns:
        np.ndarray or None: The processed image frame or None if no processing is done.
    """
    img_resized = cv2.resize(img, (W, H))
    frame = Frame(mapp, img_resized, K)
    
    if frame.id == 0:
        logging.info("No processing for the first frame.")
        return None  # Nothing to process for the first frame

    # Previous frame f2 to the current frame f1.
    if len(mapp.frames) < 2:
        logging.warning("Not enough frames to match.")
        return None  # Not enough frames to match

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    try:
        idx1, idx2, Rt = match_frames(f1, f2)
        logging.debug(f"Relative Pose (Rt): \n{Rt}")

        # Update current frame pose
        f1.pose = np.dot(Rt, f2.pose)

        # Triangulate points to get 3D coordinates in homogeneous form
        pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
        pts4d /= pts4d[:, 3:]  # Convert to Euclidean coordinates

        # Filter points based on parallax and being in front of the camera
        good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

        for i, p in enumerate(pts4d):
            if good_pts4d[i]:
                pt = Point(mapp, p[:3])
                pt.add_observation(f1, i)
                pt.add_observation(f2, i)

        # Optional: Visualize matches on the image
        for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
            u1, v1 = denormalize(K, pt1)
            u2, v2 = denormalize(K, pt2)

            cv2.circle(img_resized, (u1, v1), 2, (77, 243, 255), -1)
            cv2.line(img_resized, (u1, v1), (u2, v2), (255, 0, 0), 1)
            cv2.circle(img_resized, (u2, v2), 2, (204, 77, 255), -1)

        return img_resized

    except Exception as e:
        logging.error(f"Error during frame processing: {e}", exc_info=True)
        return None

# SLAM thread function with proper parameters and termination
def runCustomSLAM(drone, mapp, K, W, H, stop_event, frame_queue):
    """
    Runs the custom SLAM processing in a separate thread.

    Args:
        drone (Tello): The drone object.
        mapp (Map): The map object to store 3D points.
        K (np.ndarray): Camera intrinsic matrix.
        W (int): Image width.
        H (int): Image height.
        stop_event (Event): Event to signal thread termination.
        frame_queue (Queue): Queue to receive processed frames for display.
    """
    frame_read = drone.get_frame_read()

    try:
        while not stop_event.is_set():
            frame = frame_read.frame
            
            if frame is None:
                logging.warning("Received an empty frame from the drone.")
                continue  # Skip processing if the frame is None

            logging.info("Processing a new frame.")
            processed_frame = process_frame(frame, mapp, K, W, H)
            
            if processed_frame is not None:
                try:
                    frame_queue.put_nowait(processed_frame)
                except Queue.Full:
                    logging.warning("Frame queue is full. Dropping the frame.")

            sleep(0.01)  # Prevent excessive CPU usage

    except Exception as e:
        logging.error(f"Error in SLAM thread: {e}", exc_info=True)
    finally:
        try:
            mapp.export_to_csv("tmp/pointdata.csv")
            logging.info("Exported point data to 'pointdata.csv'")
        except Exception as e:
            logging.error(f"Error exporting CSV: {e}", exc_info=True)
        finally:
            drone.streamoff()
            #drone.end()

# Drone operation function with enhanced thread and error handling
def drone_mission(data):
    """
    Controls the drone to perform a 360-degree scan while running SLAM.

    Args:
        data (dict): Configuration parameters loaded from config.json.

    Returns:
        Tello: The drone object after completing the scan.
    """
    # Camera intrinsics
    W, H = 1920 // 2, 1080 // 2
    F = 450
    K = np.array([[F, 0, W // 2],
                  [0, F, H // 2],
                  [0, 0, 1]])

    # Initialize the map
    mapp = Map()

    # Initialize and connect to the drone
    try:
        drone = Tello()
        drone.connect()
        drone.speed = int(data.get("speed", 10))  # Default speed if not specified

        battery_level = drone.get_battery()
        logging.info(f"Connected to Tello Drone. Battery Level: {battery_level}%")
    except Exception as e:
        logging.error(f"Failed to connect to drone: {e}")
        return None

    # Read configuration parameters with defaults
    height = int(data.get("height", 100))  # Default height in cm
    sleepTime = int(data.get("sleep", 2))  # Default sleep time in seconds

    # Start the video stream
    try:
        drone.streamoff()
        drone.streamon()
        frame_read = drone.get_frame_read()
        logging.info("Video stream started.")
    except Exception as e:
        logging.error(f"Failed to start video stream: {e}")
        #drone.end()
        return None

    # Takeoff and move to the desired height
    try:
        drone.takeoff()
        sleep(4)
        current_height = drone.get_height()
        target_height = height
        if current_height < target_height:
            drone.move_up(target_height - current_height)
            sleep(2)  # Allow time for movement
        elif current_height > target_height:
            drone.move_down(current_height - target_height)
            sleep(2)
        logging.info(f"Drone moved to height: {target_height} cm")
    except Exception as e:
        logging.error(f"Error during takeoff/movement: {e}")
        return None

    # Set up the SLAM thread with a frame queue
    stop_event = Event()
    frame_queue = Queue(maxsize=100)  # Adjust maxsize based on memory constraints
    SLAM_THREAD = Thread(target=runCustomSLAM, args=(drone, mapp, K, W, H, stop_event, frame_queue))
    SLAM_THREAD.start()
    logging.info("SLAM thread started.")

    logging.info('Starting 360-degree rotation and SLAM processing.')

    angle =0
    # Perform a single 360-degree rotation
    try:
        while angle < MAX_ANGLE:
            logging.info("Rotating drone by 360 degrees.")
            drone.rotate_clockwise(30)
            angle += 30
            #move(drone)  # Optional up and down movement for focus
            sleep(2)

            # Handle GUI in main thread
            try:
                while True:
                    processed_frame = frame_queue.get_nowait()
                    if processed_frame is None:
                        break
                    cv2.imshow('Processed Frame', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
            except Empty:
                pass  # No frames to display at this moment

    except Exception as e:
        logging.error(f"Error during rotation: {e}")
    finally:
        # Signal the SLAM thread to stop and wait for it to finish
        stop_event.set()
        logging.info("Signaling SLAM thread to stop.")
        SLAM_THREAD.join(timeout=15)
        if SLAM_THREAD.is_alive():
            logging.warning("SLAM thread is still running after timeout.")
        else:
            logging.info("SLAM thread has finished.")

        # Handle remaining frames in the queue
        try:
            while True:
                processed_frame = frame_queue.get_nowait()
                if processed_frame is None:
                    break
                cv2.imshow('Processed Frame', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Empty:
            pass  # No more frames

        # Destroy all OpenCV windows
        cv2.destroyAllWindows()

        # Stop the video stream and land the drone
        try:
            drone.streamoff()
            #drone.land()
            logging.info("Drone has landed.")
        except Exception as e:
            logging.error(f"Error during landing or stopping stream: {e}")

    return drone

# Load the configuration from config.json with error handling
def loadConfig():
    """
    Loads the configuration parameters from config.json.

    Returns:
        dict: Configuration parameters.
    """
    try:
        with open('config.json') as f:
            config = load(f)
            logging.info("Configuration loaded successfully.")
            return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return {}

# Main execution block with comprehensive error handling
if __name__ == '__main__':
    MAX_ANGLE = 360  # Changed to 360 for full rotation

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    data = loadConfig()
    if not data:
        logging.error("No configuration data available. Exiting.")
        exit(1)

    while True:
        drone = drone_mission(data, MAX_ANGLE)
        if drone is None:
            logging.error("Drone initialization failed. Exiting.")
            break

        try:
            # Define the path to your CSV file
            csv_path = os.path.join('C:/Users/hayny/FYP/exit_autonomous', 'tmp', 'pointdata.csv')
            
            # Wait until the CSV file is created
            wait_time = 0
            while not os.path.exists(csv_path) and wait_time < 30:
                logging.info(f"Waiting for CSV file '{csv_path}' to be created...")
                sleep(1)
                wait_time += 1

            if not os.path.exists(csv_path):
                logging.error(f"CSV file not found at {csv_path} after waiting. Skipping this iteration.")
                continue

            x, y, z = readCSV(csv_path)
            pcd = makeCloud(x, y, z)
            inlierPCD, outlierPCD = removeStatisticalOutlier(
                pcd,
                voxel_size=0.01,
                nb_neighbors=30,
                std_ratio=5)
            inX, inY, inZ = pcdToArrays(inlierPCD)
            box = getAverageRectangle(inX, inZ)

            plot2DWithBox(inX, inZ, box)
            xOut, yOut = pointsOutOfBox(inX, inZ, box)
            clusters = hierarchicalClustering(xOut, yOut, 1.5)
            clustersCenters = getClustersCenters(clusters)

            # Break if there are no exits in the room
            if len(clustersCenters) == 0:
                logging.info("No exit points detected. Ending operation.")
                break
                
            print(len(clustersCenters))

            sleep(2)
            
            dronePosition = (0, 0)

            # choose the furthest exit
            maxDistance = float('-inf')
            furthestPoint = None
            for exitPoint in clustersCenters:
                distance = distanceBetween2Points(dronePosition, exitPoint)
                if distance > maxDistance:
                    maxDistance = distance
                    furthestPoint = exitPoint

            # calculate angle
            x, y = furthestPoint
            print(furthestPoint)
            angle = 90 - int(degrees(tan(float(abs(y) / abs(x)))))
            if x > 0 > y:
                angle += 90
            elif x < 0 and y < 0:
                angle += 180
            elif x < 0 < y:
                angle += 270

            drone.rotate_clockwise(angle)
            sleep(3)


            # 1 unit in ORB_SLAM2 is about 160cm in real life
            distance = int(maxDistance * 300)
            print("distance ", distance)
            while distance > 500:
                drone.move_forward(500)
                distance -= 500
            drone.move_forward(distance)
            sleep(6)


        except Exception as e:
            logging.error(f"Error during point cloud processing or drone navigation: {e}")
            break

        finally:
            try:
                drone.land()
                drone.end()
                logging.info("Drone connection ended.")
            except Exception as e:
                logging.error(f"Error during drone shutdown: {e}")