"""
Drone Detection System - Proper Version
"""

import cv2
import numpy as np
import time
import torch
import os
import csv
import datetime
from ultralytics import YOLO
from tkinter import filedialog, Tk

# ================= CONFIG =================
torch.set_num_threads(4)

CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ================= OUTPUT FOLDER =================
OUTPUT_FOLDER = "output"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Create timestamp for this run
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create CSV log file
LOG_FILE = os.path.join(OUTPUT_FOLDER, f"detection_log_{run_timestamp}.csv")

with open(LOG_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object", "Confidence"])

# ================= LOAD MODEL =================
model = YOLO("yolov8s.pt")   # Better accuracy than nano

# ================= FRAME PROCESSING =================
def process_frame(frame):

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    output = frame.copy()

    results = model(frame, imgsz=640, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):

            if confs[i] < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, xyxy[i])
            class_name = model.names[int(class_ids[i])]
            confidence = round(float(confs[i]), 2)

            label = f"{class_name} ({confidence})"

            cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(output, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2)

            # Log detection
            with open(LOG_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([class_name, confidence])

    return output

# ================= VIDEO / WEBCAM =================
def process_video(source):

    cap = cv2.VideoCapture(source)

    video_save_path = os.path.join(
        OUTPUT_FOLDER,
        f"processed_video_{run_timestamp}.avi"
    )

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        video_save_path,
        fourcc,
        20.0,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = process_frame(frame)

        current_time = time.time()
        fps = 1/(current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(output,
                    f"FPS: {int(fps)}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    2)

        cv2.imshow("Drone Detection System", output)

        out.write(output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video saved at:", video_save_path)

# ================= IMAGE MODE =================
def process_image(image_path):

    image = cv2.imread(image_path)
    output = process_frame(image)

    image_save_path = os.path.join(
        OUTPUT_FOLDER,
        f"processed_image_{run_timestamp}.jpg"
    )

    cv2.imwrite(image_save_path, output)

    print("Image saved at:", image_save_path)

    cv2.imshow("Image Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================= MENU =================
def main():

    print("\n========== DRONE DETECTION MENU ==========")
    print("1. Live Webcam")
    print("2. Insert Video")
    print("3. Insert Image")
    print("4. Exit")

    choice = input("Choose option (1-4): ")

    if choice == "1":
        process_video(0)

    elif choice == "2":
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        process_video(file_path)

    elif choice == "3":
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        process_image(file_path)

    else:
        print("Exiting.")

# ================= START =================
main()
