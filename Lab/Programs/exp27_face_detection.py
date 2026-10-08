"""
========================================================================================
EXPERIMENT 27: HUMAN FACE DETECTION USING HAAR CASCADE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement a human face detection algorithm using OpenCV's Haar Cascade Classifier
    to detect and locate faces within an input image.

THEORY & MATHEMATICAL FOUNDATION:
    The Viola-Jones face detection framework (2001) revolutionized real-time computer
    vision by combining four fundamental innovations:
    1. Haar-like Features: Rectangular filters computing difference between sum of pixels
       in adjacent regions (edge, line, center-surround).
    2. Integral Image Representation: Precomputes cumulative sums across image coordinates,
       allowing any rectangular area sum to be evaluated in constant time O(1) using only
       4 array lookups.
    3. AdaBoost Training: Selects a compact subset of highly discriminative weak classifiers
       from hundreds of thousands of candidate features.
    4. Attentional Cascade: Arranges classifiers in a multi-stage funnel where simple stages
       rapidly reject >90% of negative sub-windows, reserving complex stages for candidates.

    OpenCV API:
        faces = face_cascade.detectMultiScale(image, scaleFactor, minNeighbors, minSize)
        - scaleFactor (e.g. 1.1) : Image pyramid rescale factor per level.
        - minNeighbors (e.g. 5)  : Retains detections with >= N candidate bounding overlaps.

INPUT:
    assets/images/portrait.jpg (or group_faces.jpg)
    assets/cascades/haarcascade_frontalface_default.xml

OUTPUT:
    Side-by-side display of the input image and the localized face bounding boxes.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, get_asset_path, create_comparison, display_and_wait

def run_experiment(image_name="portrait.jpg"):
    print("[*] Running Experiment 27: Human Face Detection")

    # Step 1: Load input image
    orig_img = load_image(image_name)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Step 2: Load Haar Cascade Classifier
    cascade_path = get_asset_path("cascades", "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"[!] Error: Could not load cascade from '{cascade_path}'.")
        return

    # Step 3: Detect faces in multi-scale pyramid
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"[-] Detections: {len(faces)} human face(s) found.")

    # Step 4: Draw bounding boxes and annotations
    annotated_img = orig_img.copy()
    for idx, (x, y, w, h) in enumerate(faces):
        print(f"    -> Face {idx + 1}: Bounding Box = [X:{x}, Y:{y}, W:{w}, H:{h}]")
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(annotated_img, (x, y - 25), (x + 110, y), (0, 255, 0), -1)
        cv2.putText(annotated_img, f"Face #{idx+1}", (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Step 5: Format comparison view
    comparison = create_comparison(orig_img, annotated_img, title1="Original Image", title2=f"Detected Faces ({len(faces)})")

    # Step 6: Display result
    display_and_wait("Experiment 27 - Face Detection", comparison)

if __name__ == "__main__":
    run_experiment()
