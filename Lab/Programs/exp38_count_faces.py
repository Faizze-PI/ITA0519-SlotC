"""
========================================================================================
EXPERIMENT 38: COUNT THE NUMBER OF FACES IN AN IMAGE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to detect, enumerate, and count the total number of
    human faces in a given input image using OpenCV's Haar Cascade Classifier.

THEORY & MATHEMATICAL FOUNDATION:
    Face counting is an essential task for automated crowd monitoring, attendance
    systems, and demographic analytics.
    
    The algorithm employs the Viola-Jones object detection pipeline:
        1. Grayscale Conversion: Luminance representation for gradient-based evaluation.
        2. Multi-Scale Face Detection:
               faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        3. Statistical Tally:
               total_faces = len(faces)
        4. Geometric Annotation & Labeling:
           Iterate through the array of bounding rectangles [(x, y, w, h)]:
               - Draw bounding box for each candidate face.
               - Label each candidate with an enumerated index (Face #1, Face #2, ...).
               - Render a summary banner stating the exact total count of detected faces.

INPUT:
    assets/images/group_faces.jpg (or portrait.jpg)
    assets/cascades/haarcascade_frontalface_default.xml

OUTPUT:
    Side-by-side display of the input image and the image annotated with each face
    enumerated and a total count HUD banner.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, get_asset_path, create_comparison, display_and_wait

def count_faces_in_image(image_name="group_faces.jpg"):
    print(f"[*] Running Experiment 38: Counting Faces in Image ({image_name})")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Step 2: Load Haar Cascade
    cascade_path = get_asset_path("cascades", "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"[!] Error: Could not load cascade from '{cascade_path}'.")
        return 0

    # Step 3: Detect all faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    total_count = len(faces)

    print(f"==================================================")
    print(f"[+] TOTAL FACES DETECTED: {total_count}")
    print(f"==================================================")

    # Step 4: Annotate each face
    annotated_img = orig_img.copy()
    for idx, (x, y, w, h) in enumerate(faces):
        print(f"    Face {idx + 1}: Box at X={x}, Y={y}, Width={w}, Height={h}")
        # Bounding box
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Individual face badge
        cv2.rectangle(annotated_img, (x, y - 22), (x + 85, y), (0, 255, 0), -1)
        cv2.putText(annotated_img, f"#{idx+1}", (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Step 5: Render total count HUD header banner
    h, w = orig_img.shape[:2]
    hud = annotated_img.copy()
    cv2.rectangle(hud, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.addWeighted(hud, 0.7, annotated_img, 0.3, 0, annotated_img)
    cv2.putText(annotated_img, f"TOTAL FACES DETECTED: {total_count}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # Step 6: Create comparison view
    comparison = create_comparison(orig_img, annotated_img, title1="Original Image", title2=f"Total Faces Counted: {total_count}")

    # Step 7: Display result
    display_and_wait("Experiment 38 - Face Counting", comparison)
    return total_count

if __name__ == "__main__":
    count_faces_in_image()
