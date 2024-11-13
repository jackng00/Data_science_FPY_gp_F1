from flask import Flask, Response, request,jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import time
import math
from datetime import datetime
import base64
from ultralytics import YOLO
import cv2
import numpy as np
from vidgear.gears import CamGear
from collections import defaultdict
from djitellopy import Tello
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from collections import Counter

from Function_Area_Scanning import create_merged_map, create_separate_2d_maps,deduplicate_with_dbscan, calculate_iou, map_objects, process_frame, capture_image, define_square_movement, rotate_drone, move_drone
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RECORD_FOLDER'] = 'records/'
app.config['MODEL_FOLDER'] = 'models/'
CORS(app)

model = YOLO('model\yolov8n.pt', task='detect')

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# uri = "mongodb+srv://ngjack389:Jack9935-@dronedata.if0km.mongodb.net/?retryWrites=true&w=majority&appName=DroneData"
# # Create a new client and connect to the server
# client = MongoClient(uri, server_api=ServerApi('1'))
# db = client['filght']  # Replace 'your_database' with your database name
# collection = db['filght']  # Replace 'your_collection' with your collection name

# # Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)


#normal control
normal_control_state = True
normal_control_predict_state = False

#auto face
ALLOWED_EXTENSIONS = {'png', 'jpg', "jpeg"}
autoface_filepath = ''
autoface_state = False
autoface_takeoff = False
autoface_action = ''

#particular object
object_filepath = ''
object_state = False
object_class = ''
object_class_name= ''

#auto move
auto_move_state= False
auto_move_position = 0
auto_move_path = ''
result_data = []
frame_id = 0

#particular object functions
@app.route('/upload_object', methods=['POST'])
def upload_object():
    global object_filepath
    global object_state
    global object_class
    global object_class_name

    print( request.files['file'])
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    #with Image.open(filepath) as img:
    #   img.show()  # This will open the image in the default image viewer

    object_filepath = filepath
    object_state = True

    frame = cv2.imread(object_filepath)

    # Check if the image was loaded correctly
    if frame is None:
        print('Error loading image')
        exit()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.track(rgb_frame, conf=0.5,  tracker="bytetrack.yaml")

    for result in results:
            for obj in result:
                object_class  = int(obj.boxes.cls[0])
                object_class_name = str(obj.names[object_class])
                break

    return jsonify({'message': 'Image uploaded successfully', 'filename': filename,'class_name': object_class_name}), 200

#auto face function
@app.route('/upload', methods=['POST'])
def upload_image():
    global autoface_filepath 
    global autoface_state 

    print(request.files)
    print( request.files['file'])
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # with Image.open(filepath) as img:
    #     img.show()  # This will open the image in the default image viewer

    autoface_filepath = filepath
    autoface_state = True

    return jsonify({'message': 'Image uploaded successfully', 'filename': filename}), 200

@app.route('/face_detect_status')
def face_detect_status():
    global autoface_state
    global autoface_action
    return jsonify({'message': 'face_detect_status', 'status': autoface_state, 'action': autoface_action}), 200

@app.route('/face_detect_takeoff')
def face_detect_takeoff():
    print('face_detect_takeoff')
    global autoface_takeoff
    drone.takeoff()
    time.sleep(1)
    drone.move_up(85)
    autoface_takeoff = True
    return 'Tello has taken off and move up'

@app.route('/video_feed_autoface')
def video_feed_autoface():
    global autoface_state

    def generate_frames_auto():
        if autoface_state:
            # Load the pretrained YOLOv8 model (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
            tello = drone
            model = YOLO('./yolov8n-face.pt', task='detect')
            global autoface_filepath 

            # Create an inception resnet (in eval mode):
            resnet = InceptionResnetV1(pretrained='vggface2').eval()

            # Read and get the embedding of the target face
            #target_img_path = './uploads/mctnn_target.jpg'
            target_img_path = autoface_filepath
            target_img = Image.open(target_img_path)
            target_tensor = transforms.ToTensor()(target_img)
            target_tensor = target_tensor.unsqueeze(0)
            target_embedding = resnet(target_tensor).detach()

            # get the dimension of frame
            frame_width = 640
            frame_height = 480
            frame_x_centre = frame_width/2
            frame_y_centre = frame_height/2
            frame_area = frame_width * frame_height
            margin_x = 150
            margin_y = 80
            margin_ratio = 0.1

            countdown = {"forbackward":5, "updown":5, "rotation":5, "move":3}
            # Set Countdown for movement
            countdown_forbackward = countdown["forbackward"]
            countdown_updown = countdown["updown"]
            countdown_rotation = countdown["rotation"]
            countdown_move = countdown["move"]
            
            #while autoface_state:
            while True:

                frame = drone.get_frame_read().frame
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                #frame = stream.read()
                countdown_forbackward = countdown_forbackward-1
                countdown_rotation = countdown_rotation-1
                countdown_updown = countdown_updown-1
                countdown_move = countdown_move-1
                print('countdown:', countdown_forbackward,countdown_rotation,countdown_updown,countdown_move)

                
                if frame is not None:   #in other words, if there IS a frame...
                    # capture the starting time
                    start_time = time.time()

                    # Run YOLOv8 Object Detection on the frame
                    # 'results' saves information about the detected objects
                    results = model(frame)
                    global autoface_action
                    global autoface_takeoff
        
                    for det in results[0].boxes:
                        # det is now a single detection with attributes you can directly access
                        xmin, ymin, xmax, ymax = det.xyxy[0]  # Coordinates
                        # convert float to int
                        xmin = int(xmin)
                        ymin = int(ymin)
                        xmax = int(xmax)
                        ymax = int(ymax)

                        # get face centre and size
                        face_x_centre = int((xmax + xmin) / 2)
                        face_y_centre = int((ymax + ymin) / 2)
                        face_area = int((xmax-xmin)*(ymax-ymin))

                        # calculate the ratio face:frame, assume 1/25 is good face/frame_size ratio
                        ff_ratio = face_area/frame_area
                        #ratio_index = np.log(ff_ratio/(1/25))
                        ratio_index = (1/25)/ff_ratio
                        print(f"face_area= {face_area}, ff_ratio= {ff_ratio}, ratio_index= {ratio_index}")

                        # crop the face
                        face = frame[ymin:ymax, xmin:xmax, :]

                        # resize image according to facenet input dimensions
                        resized_face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)

                        # convert to tensor
                        resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
                        tensorImg = transforms.ToTensor()(resized_face)

                        # change the shape of tensor
                        tensorImg = tensorImg.unsqueeze(0)
                        # get the embeddings
                        embeddings_face = resnet(tensorImg).detach()

                        # Calculate the Euclidean distance between embeddings
                        distance = (target_embedding - embeddings_face).norm().item()
                        
                        print('distance',distance)
                        autoface_action = 'stay'
                        #if distance < 1.17:  # decided by the results of grid searching
                        if distance < 1.3  and autoface_takeoff == True:
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
                            cv2.putText(frame, 'Target', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            print('target')

                            # Movement check
                            if countdown_rotation<=0 and countdown_move<=0:
                                # auto-rotation
                                if face_x_centre - frame_x_centre > margin_x:
                                    #tello.rotate_clockwise(int(20*ratio_index))
                                    tello.rotate_clockwise(20)
                                    autoface_action = 'rotate clockwise'
                                    countdown_rotation = countdown["rotation"]
                                    countdown_move = countdown["move"]
                                    time.sleep(1)
                                    tello.send_rc_control(0, 0, 0, 0)
                                    print(f'rotate_clockwise, faceY:{face_y_centre}, frame_centre:{frame_y_centre}')
                                elif frame_x_centre - face_x_centre > margin_x:
                                    #tello.rotate_counter_clockwise(int(20*ratio_index))
                                    tello.rotate_counter_clockwise(20)
                                    autoface_action = 'rotate counter clockwise'
                                    countdown_rotation = countdown["rotation"]
                                    countdown_move = countdown["move"]
                                    time.sleep(1)
                                    tello.send_rc_control(0, 0, 0, 0)
                                    print(f'rotate_counter_clockwise, faceY:{face_y_centre}, frame_centre:{frame_y_centre}')

                            if countdown_updown<=0 and countdown_move<=0:
                                # auto vertical flight
                                if face_y_centre - frame_y_centre > margin_y:
                                    tello.move_down(20)
                                    autoface_action = 'move down'
                                    countdown_updown = countdown["updown"]
                                    countdown_move = countdown["move"]
                                    time.sleep(1)
                                    tello.send_rc_control(0, 0, 0, 0)
                                    print(f'move_down, faceY:{face_y_centre}, frame_centre:{frame_y_centre}')
                                if frame_y_centre - face_y_centre > margin_y:
                                    tello.move_up(20)
                                    autoface_action = 'move up'
                                    countdown_updown = countdown["updown"]
                                    countdown_move = countdown["move"]
                                    time.sleep(1)
                                    tello.send_rc_control(0, 0, 0, 0)
                                    print(f'move_up, faceY:{face_y_centre}, frame_centre:{frame_y_centre}')

                            if countdown_forbackward<=0 and countdown_move<=0:
                                # auto forward-backward flight
                                if ff_ratio > 1/15:
                                    #tello.move_backward(int(20*ratio_index))
                                    #tello.move_backward(20)
                                    tello.move_back(40)
                                    autoface_action = 'move back'
                                    countdown_forbackward = countdown["forbackward"]
                                    countdown_move = countdown["move"]
                                    time.sleep(1)
                                    tello.send_rc_control(0, 0, 0, 0)
                                    print(f'move_backward, faceY:{face_y_centre}, frame_centre:{frame_y_centre}')
                                if ff_ratio < 1/40:
                                    #tello.move_forward(int(20*ratio_index))
                                    tello.move_forward(min(int(35*ratio_index),50))
                                    autoface_action = 'move forward'
                                    countdown_forbackward = countdown["forbackward"]
                                    countdown_move = countdown["move"]
                                    time.sleep(1)
                                    tello.send_rc_control(0, 0, 0, 0)
                                    print(f'move_forward, faceY:{face_y_centre}, frame_centre:{frame_y_centre}')
                        else:
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)


                    end_time = time.time()
                    execution_time = end_time - start_time
                    #ret, buffer = cv2.imencode('.jpg', frame)
                    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
    return Response(generate_frames_auto(), mimetype='multipart/x-mixed-replace; boundary=frame')

#view for particular object 
@app.route('/video_feed_object')
def video_feed_object():
    print('video')
    global object_class_name
    global object_state

    def generate_frames():
        while object_state:
            frame = drone.get_frame_read().frame
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #frame = stream.read()
            results = model(frame)  

            #Visualize the results on the frame
            for result in results:
                for obj in result:
                    object_class_result = int(obj.boxes.cls[0])
                    object_class_name_result = str(obj.names[object_class_result])
                    #print('object_class_name_result',object_class_name_result)
                    #print('object_class_name',object_class_name)

                    # Filter to only plot objects with the class name "chair"
                    if object_class_name_result == object_class_name:
                        x1, y1, x2, y2 = map(int, obj.boxes.xyxy[0])
                        confidence = obj.boxes.conf[0]
                        label = f'{object_class_name} {confidence:.2f}'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert the annotated frame colors from BGR to RGB
            #annotated_frame = cv2.cvtColor(annotated_frame,cv2.COLOR_BGR2RGB)

            #ret, buffer = cv2.imencode('.jpg', frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#normal view
@app.route('/video_feed')
def video_feed():
    print('video')
    def generate_frames():
        while normal_control_state:
            frame = drone.get_frame_read().frame
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #frame = stream.read()
            if normal_control_predict_state:
                results = model(frame)  

                #Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Convert the annotated frame colors from BGR to RGB
                #annotated_frame = cv2.cvtColor(annotated_frame,cv2.COLOR_BGR2RGB)

            if normal_control_predict_state:
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#auto_move view
@app.route('/video_feed_auto_move')
def video_feed_auto_move():
    global auto_move_state
    print('video')
    def generate_frames():
        while auto_move_state:
            frame = drone.get_frame_read().frame
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # Convert the annotated frame colors from BGR to RGB
            annotated_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            #ret, buffer = cv2.imencode('.jpg', frame)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/auto_move_new')
def auto_move_new():
    # Initialize Object Map
    object_map = {}

    # Initialize Drone State
    drone_state = DroneState()

    image_folder = "scan_image"
    #os.makedirs(image_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Initialize Drone
    tello = drone
    try:
        #tello.connect()
        battery_level = tello.get_battery()
        print(f"Connected to Tello Drone. Battery Level: {battery_level}%")

        if battery_level < 20:
            print("Battery too low. Please charge the drone before starting.")
            return

        # Enable SDK Live Video
        #tello.streamon()

        # Load YOLO Model
        yolo_model = YOLO('model/yolo11n.pt', task='detect')
        print('load yolo_model ok')

        # Load MiDaS Model
        model_type = "DPT_Large"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, verbose=False)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        print('load DPT_Large ok')
        # Load MiDaS Transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        # Define Square Movement Parameters
        #side_length = 450  # cm
        side_length = 60  # cm
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
        #drone.send_rc_control(0, 0, 0, 0)
        time.sleep(2)  # Wait for drone to stabilize in the air
        #tello.move_up(100)
        tello.move_up(70)
        #drone.send_rc_control(0, 0, 0, 0)
        time.sleep(2)
        print("Takeoff complete.")
        print("Starting flight and data capture...")

        ##First Capture to initiate
        frame = tello.get_frame_read().frame
        global auto_move_position
        global auto_move_path
        count = 0

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
                    #cv2.imshow(f"Waypoint {idx} - Annotated Frame", annotated_frame)
                    
                    # Save the annotated image
                    image_filename = os.path.join(image_folder, f"waypoint_{idx}_capture_raw.jpg")
                    cv2.imwrite(image_filename, frame)
                    
                    output_image_path = os.path.join(image_folder, f"waypoint_{idx}_capture_annotated.jpg")
                    annotated_output_image_path = output_image_path
                    cv2.imwrite(output_image_path, annotated_frame)
                    print(f"Annotated image saved to {output_image_path}.")
                    
                    output_image_path = os.path.join(image_folder, f"waypoint_{idx}_capture_depth.jpg")
                    cv2.imwrite(output_image_path, depth_frame)
                    print(f"depth image saved to {output_image_path}.")

                    count = count +1
                    auto_move_position = count
                    auto_move_path = annotated_output_image_path

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
        create_separate_2d_maps(object_map,'scan_image/separate_2d_maps.png')
        print("Separate 2D object maps generation complete.")

        print("\nGenerating merged 2D object map...")
        create_merged_map(object_map,'scan_image/merged_2d_map.png')
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
        
        auto_move_position = 5
        # Cleanup
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()
        print("Exiting...")

@app.route('/auto_move_status')
def auto_move_status():
    global auto_move_position
    global auto_move_path
    if auto_move_position > 0 and auto_move_position <=4:
        return jsonify({'message': 'auto_move_status', 'status': auto_move_position, 'path': auto_move_path}), 200
    elif auto_move_position == 5:
        return jsonify({'message': 'auto_move_status_finish', 'status': auto_move_position, 'path1': 'scan_image/separate_2d_maps.png', 'path2': 'scan_image/merged_2d_map.png'}), 200
    return jsonify({'message': 'auto_move_status_start', 'status': auto_move_position}), 200
    

#normal control
@app.route('/takeoff')
def takeoff():
    print('takeoff')
    drone.takeoff()
    return 'Tello has taken off'

@app.route('/land')
def land():
    print('land')
    drone.land()
    return 'Tello is landing'

@app.route('/move')
def move():
    print('move')
    direction = request.args.get('direction')
    distance = int(request.args.get('distance'))
    if direction == 'forward':
        drone.send_rc_control(0, distance, 0, 0)
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)

    if direction == 'backward':
        drone.send_rc_control(0, -distance, 0, 0)
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)

    if direction == 'right':
        drone.send_rc_control(distance, 0, 0, 0)
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)

    if direction == 'left':
        drone.send_rc_control(-distance, 0, 0, 0)
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)
    
    if direction == 'up':
        drone.move_up(distance)
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)

    if direction == 'down':
        drone.move_down(distance)
        time.sleep(2)
        drone.send_rc_control(0, 0, 0, 0)

    return f'Tello is moving {direction} by {distance} cm'

@app.route('/rotate')
def rotate():
    print('rotate')
    direction = request.args.get('direction')
    degree = int(request.args.get('degree'))
    speed = 30
    rotation_time = abs(degree) / speed

    if direction == 'clockwise':
        drone.rotate_clockwise(degree)

    if direction == 'counter-clockwise':
        drone.rotate_counter_clockwise(degree)
    
    time.sleep(rotation_time)
    drone.send_rc_control(0, 0, 0, 0)

    return f'Tello is rotating {direction} by {degree} degree'

@app.route('/change_normal_predict_state')
def change_normal_predict_state():
    global normal_control_predict_state
    normal_control_predict_state = not normal_control_predict_state
    print('change_normal_predict_state', normal_control_predict_state)
    return f'Changed normal_control_predict_state to {normal_control_predict_state}'

@app.route('/one_time_predict', methods=['POST'])
def one_time_predict():
    with app.app_context():
        one_time_result_data = []
        frame = drone.get_frame_read().frame
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #frame = stream.read()
        results = model(frame)  
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame,cv2.COLOR_BGR2RGB)

        for result in results:
            for obj in result:
                # Extract bounding box coordinates and class info
                x1, y1, x2, y2 = map(int, obj.boxes.xyxy[0])
                class_id = int(obj.boxes.cls[0])
                class_name = str(obj.names[class_id])
                confidence = float(obj.boxes.conf[0])
                track_id = obj.boxes.id

                # Append data for each detection
                one_time_result_data.append({
                    'task': 'one_time_detect',
                    'frame_id': frame_id,
                    'class_id': class_id,
                    'class_name': class_name,
                    'track_id': track_id,
                    'confidence': confidence,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })

        #collection.insert_many(one_time_result_data)

        output_path = save_image(annotated_frame)
        #print('encoded_string',encoded_string)
        return jsonify({"image_path": output_path,"result": summarize_classes(one_time_result_data) }), 200

@app.route('/get_image/<path:filename>', methods=['GET'])
def get_image(filename):
    return send_file(filename, mimetype='image/jpeg')

def summarize_classes(data): 
    class_counter = Counter(item['class_name'] for item in data) 
    summary = ', '.join(f'{cls}: {count}' for cls, count in class_counter.items()) 
    return summary
    
def save_image(annotated_frame):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the frame to an image file with the timestamp in the filename
    output_filename = f'annotated_frame_{current_time}.jpg'
    output_path = os.path.join(app.config['RECORD_FOLDER'], output_filename)

    # Save the image
    cv2.imwrite(output_path, annotated_frame)

    print(f"Frame saved to {output_path}")
    return output_path


@app.route('/change_state')
def change_state():
    print('change_state')
    state = request.args.get('state')

    global normal_control_state
    global autoface_state
    global object_state
    global auto_move_state

    normal_control_state = False
    autoface_state = False
    object_state = False
    auto_move_state= False

    if state == 'normal':
        normal_control_state = True

    if state == 'auto_face':
        autoface_state = True
    
    if state == 'object':
        object_state = True
    
    if state == 'auto_move':
        auto_move_state = True
    
    return f'Changed state to {state}'

@app.route('/select_model', methods=['POST'])
def select_model():
    data = request.get_json()
    model_name = data.get('model_name')
    global model
    
    if model_name and model_name in os.listdir(app.config['MODEL_FOLDER']):
        # Load the selected model (this is just an example, adjust as needed)
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
        # Assuming you have a function to load your model
        model = YOLO(model_path, task='detect')
        return jsonify({"message": f"Model {model_name} loaded successfully."})
    else:
        return jsonify({"error": "Model not found."}), 404


if __name__ == '__main__':
    drone = Tello()
    drone.connect()
    drone.streamon()
    #stream = CamGear(source='https://www.youtube.com/watch?v=PPlnXrV-Xc4&pp=ygUZZHJvbmUgY2FtZXJhIGNhciBhY2NpZGVudA%3D%3D', stream_mode=True, logging=True).start() # YouTube Video URL as input
    
    app.run(port=5000)