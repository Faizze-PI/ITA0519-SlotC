"""
========================================================================================
EXPERIMENT 28: VEHICLE DETECTION IN VIDEO
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement a vehicle detection algorithm using OpenCV to detect and locate moving
    vehicles in each sequential frame of a video stream.

THEORY & MATHEMATICAL FOUNDATION:
    Vehicle detection in video feeds is a cornerstone of Intelligent Transportation
    Systems (ITS). In OpenCV, this is solved through two classical paradigms:
    1. Haar Feature Cascade Classifiers:
       Pretrained models (e.g. cars.xml) detect rigid car structures (windshields, wheels,
       grilles) using rectangular Haar-like features and AdaBoost cascades.
    2. Gaussian Mixture Model Background Subtraction (MOG2):
       Models background pixel intensity distributions dynamically:
           P(x_t) = sum_{k=1}^K w_{k,t} * eta(x_t, mu_{k,t}, Sigma_{k,t})
       Subtracting the background isolates moving vehicles as connected binary blobs.
       Contours with area > MinArea are bounded with rectangles.

    This program implements an intelligent detector combining Haar Cascade detection
    with MOG2 motion verification for robust real-time tracking across all video frames.

INPUT:
    assets/videos/sample_traffic.mp4
    assets/cascades/haarcascade_cars.xml

OUTPUT:
    Real-time video playback window with detected vehicles bounded in green rectangles.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import get_asset_path

def run_experiment(video_name="sample_traffic.mp4"):
    print("[*] Running Experiment 28: Vehicle Detection in Video")
    video_path = get_asset_path("videos", video_name)
    cascade_path = get_asset_path("cascades", "haarcascade_cars.xml")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video at '{video_path}'.")
        return

    # Initialize Car Cascade
    car_cascade = cv2.CascadeClassifier(cascade_path)
    use_cascade = not car_cascade.empty()
    if use_cascade:
        print("[+] Haar Car Cascade loaded successfully.")
    else:
        print("[!] Note: Haar car cascade not available; utilizing MOG2 background subtractor.")

    # Initialize MOG2 Subtractor as backup / motion booster
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

    window_name = "Experiment 28 - Real-Time Vehicle Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop for demonstration
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret: break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        vehicle_boxes = []

        # Strategy A: Cascade Detection
        if use_cascade:
            cars = car_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=(40, 40))
            for (x, y, w, h) in cars:
                vehicle_boxes.append((x, y, w, h))

        # Strategy B: Background Subtraction fallback/enrichment
        if len(vehicle_boxes) == 0:
            fg_mask = bg_subtractor.apply(frame)
            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 900: # Threshold for vehicle size
                    x, y, w, h = cv2.boundingRect(cnt)
                    vehicle_boxes.append((x, y, w, h))

        # Draw detected vehicles
        for idx, (x, y, w, h) in enumerate(vehicle_boxes):
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Vehicle", (x, max(20, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw status HUD
        cv2.rectangle(display_frame, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, f"Vehicles Tracked: {len(vehicle_boxes)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(40) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[-] Vehicle detection completed.")

if __name__ == "__main__":
    run_experiment()
