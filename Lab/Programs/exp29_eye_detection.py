"""
========================================================================================
EXPERIMENT 29: HUMAN EYE DETECTION USING HAAR CASCADE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement an eye detection algorithm using OpenCV's Haar Cascade Classifiers to
    detect and locate human eyes within facial regions of an input image.

THEORY & MATHEMATICAL FOUNDATION:
    Detecting eyes in a full unconstrained scene directly is prone to high false-positive
    rates due to repetitive textures in clothing and background clutter.
    
    The standard hierarchical computer vision pipeline solves this using a two-stage cascade:
    1. Primary Stage (Face ROI Localization):
       A frontal face cascade detects the overall head boundary (x, y, w, h).
    2. Secondary Stage (Constrained Eye Search):
       The eye cascade search space is constrained strictly within the upper half of the
       face ROI (y to y + int(0.6 * h)), since anatomically eyes reside in the upper cranial half.

    OpenCV API:
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

INPUT:
    assets/images/portrait.jpg
    assets/cascades/haarcascade_frontalface_default.xml
    assets/cascades/haarcascade_eye.xml

OUTPUT:
    Side-by-side display of the original image and the image annotated with blue eye boxes.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, get_asset_path, create_comparison, display_and_wait

def run_experiment(image_name="portrait.jpg"):
    print("[*] Running Experiment 29: Human Eye Detection")

    # Step 1: Load input image
    orig_img = load_image(image_name)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Step 2: Load Haar Cascades for Face and Eyes
    face_cascade_path = get_asset_path("cascades", "haarcascade_frontalface_default.xml")
    eye_cascade_path = get_asset_path("cascades", "haarcascade_eye.xml")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty() or eye_cascade.empty():
        print("[!] Error: Could not load required cascade XML files.")
        return

    # Step 3: Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    print(f"[-] Located {len(faces)} face(s) for localized eye search.")

    annotated_img = orig_img.copy()
    total_eyes_detected = 0

    # Step 4: Constrain eye detection within face ROIs
    for (fx, fy, fw, fh) in faces:
        # Draw green box on face
        cv2.rectangle(annotated_img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        
        # Upper face region for eyes
        eye_region_h = int(fh * 0.6)
        roi_gray_eyes = gray[fy:fy + eye_region_h, fx:fx + fw]
        roi_color_eyes = annotated_img[fy:fy + eye_region_h, fx:fx + fw]

        # Detect eyes within upper face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray_eyes, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
        total_eyes_detected += len(eyes)

        for (ex, ey, ew, eh) in eyes:
            # Draw blue bounding box on each detected eye
            cv2.rectangle(roi_color_eyes, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.circle(roi_color_eyes, (ex + ew // 2, ey + eh // 2), 3, (0, 255, 255), -1)

    print(f"[-] Total eyes detected: {total_eyes_detected}")

    # Step 5: Format comparison view
    comparison = create_comparison(orig_img, annotated_img, title1="Original Portrait", title2=f"Detected Eyes ({total_eyes_detected})")

    # Step 6: Display result
    display_and_wait("Experiment 29 - Eye Detection", comparison)

if __name__ == "__main__":
    run_experiment()
