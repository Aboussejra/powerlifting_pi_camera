import cv2
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter

min_conf_threshold = 0.5

MODEL_NAME = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME)

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Testing if quantized or not
floating_model = (input_details[0]['dtype'] == np.float32)

debug = True

output_stride = 32
input_mean = 127.5
input_std = 127.5

def mod(a, b):
    """find a % b"""
    floored = np.floor_divide(a, b)
    return np.subtract(a, np.multiply(floored, b))

def sigmoid(x):
    """apply sigmoid actiation to numpy array"""
    return 1/ (1 + np.exp(-x))

def sigmoid_and_argmax2d(inputs, threshold):
    """return y,x coordinates from heatmap"""
    #v1 is 9x9x17 heatmap
    v1 = interpreter.get_tensor(output_details[0]['index'])[0]
    height = v1.shape[0]
    width = v1.shape[1]
    depth = v1.shape[2]
    reshaped = np.reshape(v1, [height * width, depth])
    reshaped = sigmoid(reshaped)
    #apply threshold
    reshaped = (reshaped > threshold) * reshaped
    coords = np.argmax(reshaped, axis=0)
    yCoords = np.round(np.expand_dims(np.divide(coords, width), 1))
    xCoords = np.expand_dims(mod(coords, width), 1)
    return np.concatenate([yCoords, xCoords], 1)

def get_offset_point(y, x, offsets, keypoint, num_key_points):
    """get offset vector from coordinate"""
    y_off = offsets[y,x, keypoint]
    x_off = offsets[y,x, keypoint+num_key_points]
    return np.array([y_off, x_off])


def get_offsets(output_details, coords, num_key_points=17):
    """get offset vectors from all coordinates"""
    offsets = interpreter.get_tensor(output_details[1]['index'])[0]
    offset_vectors = np.array([]).reshape(-1,2)
    for i in range(len(coords)):
        heatmap_y = int(coords[i][0])
        heatmap_x = int(coords[i][1])
        #make sure indices aren't out of range
        if heatmap_y >8:
            heatmap_y = heatmap_y -1
        if heatmap_x > 8:
            heatmap_x = heatmap_x -1
        offset_vectors = np.vstack((offset_vectors, get_offset_point(heatmap_y, heatmap_x, offsets, i, num_key_points)))
    return offset_vectors

def draw_lines(keypoints, image, bad_pts):
    """connect important body part keypoints with lines"""
    #color = (255, 0, 0)
    color = (0, 255, 0)
    thickness = 2
    #refernce for keypoint indexing: https://www.tensorflow.org/lite/models/pose_estimation/overview
    body_map = [[5,6], [5,7], [7,9], [5,11], [6,8], [8,10], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]]
    for map_pair in body_map:
        #print(f'Map pair {map_pair}')
        if map_pair[0] in bad_pts or map_pair[1] in bad_pts:
            continue
        start_pos = (int(keypoints[map_pair[0]][1]), int(keypoints[map_pair[0]][0]))
        end_pos = (int(keypoints[map_pair[1]][1]), int(keypoints[map_pair[1]][0]))
        image = cv2.line(image, start_pos, end_pos, color, thickness)
    return image


def predict_frame(frame):
    t1 = cv2.getTickCount()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    #get y,x positions from heatmap
    coords = sigmoid_and_argmax2d(output_details, min_conf_threshold)
    #keep track of keypoints that don't meet threshold
    drop_pts = list(np.unique(np.where(coords ==0)[0]))
    #get offets from postions
    offset_vectors = get_offsets(output_details, coords)
    #use stide to get coordinates in image coordinates
    keypoint_positions = coords * output_stride + offset_vectors

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(keypoint_positions)):
        #don't draw low confidence points
        if i in drop_pts:
            continue
        # Center coordinates
        x = int(keypoint_positions[i][1])
        y = int(keypoint_positions[i][0])
        center_coordinates = (x, y)
        radius = 2
        color = (0, 255, 0)
        thickness = 2
        cv2.circle(frame_resized, center_coordinates, radius, color, thickness)
        if debug:
            cv2.putText(frame_resized, str(i), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1) # Draw label text

    frame_resized = draw_lines(keypoint_positions, frame_resized, drop_pts)
    return frame_resized
