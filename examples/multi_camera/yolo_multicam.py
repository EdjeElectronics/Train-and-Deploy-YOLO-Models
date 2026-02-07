##### YOLO Multiple Camera Example #####

# Author: Evan Juras, EJ Technology Consultants, https://ejtech.io

# Description:
# This script runs YOLO detection on multiple camera sources simultaneously. It uses the Python multiprocessing 
# module to create separate subprocesses for running inference on each camera. Each subprocess grabs frames from
# its camera, runs inference, and displays the detection results in a continuous loop until it is stopped.

# Import necessary packages
from multiprocessing import Process
import cv2
import os
import numpy as np
from collections import deque
from ultralytics import YOLO
import time


# Globals
bbox_colors = [(167,121,78),(43,142,242),(89,87,225),(178,183,118),(79,161,89),
    (72,201,237),(161,122,176),(167,157,255),(95,117,156),(175,176,186)]

# Define function to run inference on a camera source
def inference_camera(source, model_fn, process_id):

    # Set path to model
    cwd = os.getcwd()
    model_fn = model_fn
    model_path = os.path.join(cwd,model_fn)

    # Set program parameters
    imgW, imgH = 640, 480
    min_conf_threshold = 0.5

    # Set up buffer for frame rate calculation
    frame_rate_calcs = deque([],maxlen=100) # Create queue for holding FPS calculations
    frame_rate_avg = 0

    ### Load a model and labels
    model = YOLO(model_path)  # pretrained YOLOv8 model
    labels = model.names

    ### Load video file and process every frame of video
    cam = cv2.VideoCapture(source)

    while(cam.isOpened()):

        # Start timer for calculating framerate
        t_start = time.perf_counter()

        ret, frame = cam.read()
        if not ret or len(frame) == 0:
            print(f'Unable to access camera {source}! This likely indicates a problem with the camera.')
            break

        # Resize frame, run inference, extract results
        frame = cv2.resize(frame, (imgW, imgH))
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Reset variable for basic object counting example
        object_count = 0

        # Go through each detection and get bbox coords, confidence, and class
        for i in range(len(detections)):

            # Get bounding box coordinates
            # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
            xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # Get bounding box confidence
            conf = detections[i].conf.item()

            # Draw box if confidence threshold is high enough
            if conf > min_conf_threshold:

                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

                # Basic example: count the number of objects in the image
                object_count = object_count + 1


        cv2.putText(frame, f'FPS: {frame_rate_avg:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate
        cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
        cv2.imshow(f'Camera {source} detections (Process {process_id})', frame)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        
        # Stop framerate timer
        t_stop = time.perf_counter()
        t_total = t_stop - t_start
        frame_rate_calcs.appendleft(1/t_total)
        frame_rate_avg = np.mean(frame_rate_calcs)
        
    cv2.destroyAllWindows()
    return

# Entry point for main program
if __name__ == '__main__':
    
    # Define list of camera sources (by USB index) and models to use
    sources = [0, 1, 2, 3]
    models = ['yolo11s.pt', 'yolo11s.pt', 'yolo11s.pt', 'yolo11s.pt']
    
    # Create a new process for each camera source. The process targets the "inference_camera" function.
    # Basically, for each camera, it will open a new Python subprocess that runs the "inference_camera" function,
    # which will grab frames from the camera, run inference on them, and display results.
    procs = []
    for i in range(len(sources)):
        process = Process(target=inference_camera, args=(sources[i], models[i], i))
        process.start()
        procs.append(process)

    # Wait for processes to finish (which will not happen until the user presses 'q' on each display window)
    for p in procs:
        p.join()
