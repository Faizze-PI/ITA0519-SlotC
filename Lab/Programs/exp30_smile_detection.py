"""
========================================================================================
EXPERIMENT 30: HUMAN SMILE DETECTION USING HAAR CASCADE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement a smile detection algorithm using OpenCV's Haar Cascade Classifiers to
    detect and locate human smiles within facial regions in an input image.

THEORY & MATHEMATICAL FOUNDATION:
    Smile detection relies on tracking characteristic morphological deformations around
    the mouth: lip curvature, exposed teeth luminance, and cheek compression.
    
    Hierarchical Cascading Pipeline:
    1. Face Localization: Detects full face bounding box (x, y, w, h).
    2. Lower Facial Masking: Constrains smile search to the lower third/half of the face:
           roi_mouth = gray[y + int(0.5 * h) : y + h, x : x + w]
    3. Tuned Multi-Scale Detection:
       The smile cascade (haarcascade_smile.xml) is inherently sensitive to subtle mouth
       features; therefore, higher minNeighbors (e.g. 18 to 22) and larger scaleFactor (e.g. 1.7)
       are employed to suppress neutral mouth expressions and prevent false positives.

INPUT:
    assets/images/portrait.jpg
    assets/cascades/haarcascade_frontalface_default.xml
    assets/cascades/haarcascade_smile.xml

OUTPUT:
    Side-by-side display of the input image and the detected smile bounding box.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, get_asset_path, create_comparison, display_and_wait

def run_experiment(image_name="portrait.jpg"):
    print("[*] Running Experiment 30: Human Smile Detection")

    # Step 1: Load input image
    orig_img = load_image(image_name)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Step 2: Load Haar Cascades for Face and Smile
    face_cascade_path = get_asset_path("cascades", "haarcascade_frontalface_default.xml")
    smile_cascade_path = get_asset_path("cascades", "haarcascade_smile.xml")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

    if face_cascade.empty() or smile_cascade.empty():
        print("[!] Error: Could not load required cascade XML files.")
        return

    # Step 3: Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    print(f"[-] Located {len(faces)} face(s) for localized smile search.")

    annotated_img = orig_img.copy()
    total_smiles_detected = 0

    # Step 4: Search for smile within lower face region
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(annotated_img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        # Lower face ROI for mouth/smile
        y_mouth_start = fy + int(fh * 0.5)
        roi_gray_mouth = gray[y_mouth_start:fy + fh, fx:fx + fw]
        roi_color_mouth = annotated_img[y_mouth_start:fy + fh, fx:fx + fw]

        # Detect smiles with tuned threshold
        smiles = smile_cascade.detectMultiScale(roi_gray_mouth, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
        total_smiles_detected += len(smiles)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color_mouth, (sx, sy), (sx + sw, sy + sh), (0, 165, 255), 2)
            cv2.putText(roi_color_mouth, "SMILE", (sx, sy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    print(f"[-] Total smiles detected: {total_smiles_detected}")

    # Step 5: Format comparison view
    comparison = create_comparison(orig_img, annotated_img, title1="Original Portrait", title2=f"Smile Detection (Count: {total_smiles_detected})")

    # Step 6: Display result
    display_and_wait("Experiment 30 - Smile Detection", comparison)

if __name__ == "__main__":
    run_experiment()
