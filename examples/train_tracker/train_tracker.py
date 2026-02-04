##### TrainTracker #####

# Author: Evan Juras, EJ Technology Consultants, https://ejtech.io

# Description:
# This program uses a custom YOLO detection model to continuously watch for passing trains. When a train 
# passes, it counts the number and type of each car in the train, and then sends an email with information
# about the train. It also records a video of the train and saves it on disk.

# Import necessary packages
import cv2
import time
import smtplib
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO

### User-defined variables
model_path = 'yolo26s_train_tracker.pt'
cam_idx = 0 # USB camera index
#cam_idx = 'test1.mp4' # Uncomment to test on a video file instead of camera
conf_thresh = 0.5
train_timeout_sec = 10  # Time to wait after last car to send email
record_video = True

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Email Config
smtp_server = 'smtp.gmail.com'
smtp_port = 587
sender_email = 'my_email@gmail.com'
sender_pw = 'my app password'
recipient_email = 'receiver_email@gmail.com'

### Define function to send email
def send_summary_email(train_data, start_time, direction):
    print(" Composing email...")
    
    total_cars = sum(train_data.values())
    formatted_time = start_time.strftime("%I:%M%p, %b %d %Y")
    
    body = f"Train Report\n"
    body += f"Direction: {direction}\n"
    body += f"Time Sighted: {formatted_time}\n"
    body += f"Total Cars: {total_cars}\n\n"
    body += "Breakdown by Type:\n"
    
    for car_type, count in train_data.items():
        body += f"- {car_type}: {count}\n"
    
    print(body)

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"{direction} Train Detected: {total_cars} cars"
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_pw)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        print('Email sent successfully!')
    except Exception as e:
        print(f'Failed to send email: {e}')

### Main application logic
def main():

    # Load YOLO model
    model = YOLO(model_path)
    class_names = model.names 
    
    # Initialize camera and set resolution
    frameW, frameH = 1280, 720
    cap = cv2.VideoCapture(cam_idx)
    ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameW)
    ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0

    # State Variables
    train_is_passing = False
    last_detection_time = 0
    train_start_time = None
    
    # Tracking Data
    previous_centers = {} # Stores {track_id: x_position}
    car_counts = defaultdict(int)
    
    # Direction Variables
    line_x_position = int(frameW / 2)
    line_color = (0, 0, 255) 
    direction_score = 0 
    current_direction_str = "Unknown"
    print('Starting TrainTracker main loop.')

    # Video Recording
    video_writer = None
    current_video_filename = ""

    ### Continuously grab frames from camera, inference them, and run tracking logic
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Unable to connect to camera or reached end of test video. Exiting program.')
            break

        write_frame = np.copy(frame)
        current_time = time.time()
        
        # Run YOLO detection model with tracking enabled)
        results = model.track(frame, persist=True, verbose=False, conf=conf_thresh)[0]
        
        # Extract results
        if results.boxes.id is not None:
            last_detection_time = current_time
            
            boxes = results.boxes.xyxy.cpu()
            class_idxs = results.boxes.cls.cpu().tolist()
            track_ids = results.boxes.id.int().cpu().tolist()

            # If we are not in "train_is_passing" state and an object has been detected, it must be a train! Start a new train event.
            if not train_is_passing:
                print("Train detected! Recording started.")
                train_is_passing = True
                train_start_time = datetime.now()
                car_counts = defaultdict(int)
                previous_centers = {}
                direction_score = 0
                current_direction_str = "Calculating..."

                # Initialize video writer to record new video
                if record_video:
                    timestamp_str = train_start_time.strftime("%Y%m%d_%H%M%S")
                    current_video_filename = f"TrainTracker_{timestamp_str}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(current_video_filename, fourcc, fps, (frameW, frameH))


            # Go through each detection result and process it
            for box, classidx, track_id in zip(boxes, class_idxs, track_ids):
                x1, y1, x2, y2 = box
                xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)
                
                classname = class_names.get(int(classidx), "Unknown")

                ### Tracking and counting logic
                if track_id in previous_centers:
                    prev_x = previous_centers[track_id]
                    
                    # Calculate movement delta for direction detection
                    dx = center_x - prev_x
                    direction_score += dx # Positive = Moving Right, Negative = Moving Left

                    # Check line crossing: increment count whenever an object crosses the center line
                    # Case 1: Moving Left -> Right (Eastbound)
                    if prev_x < line_x_position and center_x >= line_x_position:
                        car_counts[classname] += 1
                        line_color = (0, 255, 0) # Flash Green
                        print(f"   -> Counted {classname} (Eastbound)")
                        
                    # Case 2: Moving Right -> Left (Westbound)
                    elif prev_x > line_x_position and center_x <= line_x_position:
                        car_counts[classname] += 1
                        line_color = (0, 255, 0) # Flash Green
                        print(f"   <- Counted {classname} (Westbound)")

                # Update storage for next frame
                previous_centers[track_id] = center_x

                # Draw bounding box and center dot
                color = bbox_colors[int(classidx) % 10]
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                label = f'{classname}:{track_id}'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

                cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)


                # Determine Direction String based on accumulated score
                if abs(direction_score) > 10: # Threshold to avoid noise
                    if direction_score > 0:
                        current_direction_str = "Eastbound"
                    else:
                        current_direction_str = "Westbound"

        # Draw the Counting Line
        cv2.line(frame, (line_x_position, 0), (line_x_position, frameH), line_color, 2)

        # If it's been long enough since the train has finished passing, stop video recoring and send an email
        if train_is_passing and (current_time - last_detection_time > train_timeout_sec):
            print(f"Train passed ({current_direction_str}). Saving video and sending email...")

            if record_video:
                    video_writer.release()
                    video_writer = None

            # Clean up direction string for email
            if current_direction_str in ["Unknown", "Calculating..."]:
                final_dir = "Unknown Direction"
            else:
                final_dir = current_direction_str # "Eastbound" or "Westbound"
            
            send_summary_email(car_counts, train_start_time, final_dir)
            
            train_is_passing = False
            previous_centers = {} # Clear tracking history
            print("Returning to idle state...")

        ### Draw Train Information on Right Sidebar

        # Create a black image with space for drawing information on the right
        total_h = frameH
        total_w = frameW + 350
        combined_img = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        combined_img[0:frameH, 0:frameW] = frame # Copy existing frame into new combined image

        # Variables for text positioning
        ui_x = frameW + 10
        ui_y = 40

        # Draw recording status
        status_text = "Recording" if train_is_passing else "Waiting"
        status_color = (0, 0, 255) if train_is_passing else (0, 255, 0)
        cv2.putText(combined_img, "STATUS:", (ui_x, ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined_img, status_text, (ui_x, ui_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        ui_y += 80

        if train_is_passing:
            # Direction
            cv2.putText(combined_img, "DIRECTION:", (ui_x, ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_img, current_direction_str, (ui_x, ui_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            ui_y += 70
            
            # Total Count
            total = sum(car_counts.values())
            cv2.putText(combined_img, f"TOTAL CARS: {total}", (ui_x, ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ui_y += 20
            
            # Divider
            cv2.line(combined_img, (frameW, ui_y), (total_w, ui_y), (100, 100, 100), 1)
            ui_y += 30
            
            # List Breakdown
            for cls_name, count in car_counts.items():
                if count > 0:
                    text = f"{cls_name}: {count}"
                    cv2.putText(combined_img, text, (ui_x, ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2)
                    ui_y += 25

        cv2.imshow("TrainTracker Feed", combined_img)
        if record_video and video_writer is not None:
            video_writer.write(write_frame) # Write original unlabeled frame to video

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    if record_video: video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
