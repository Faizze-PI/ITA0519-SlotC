"""
========================================================================================
EXPERIMENT 11: 180-DEGREE ROTATION CLOCKWISE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform a 180-degree clockwise rotation on a given image using Python and OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    In 2D digital image processing, a 180-degree rotation inverts both spatial axes:
        x' = W - 1 - x
        y' = H - 1 - y
    This produces an upside-down, inverted version of the original image without altering
    the aspect ratio or bounding dimension shape.

    OpenCV provides an optimized hardware-accelerated function:
        cv2.rotate(src, cv2.ROTATE_180)
    (Equivalently achieved by flipping both axes simultaneously: cv2.flip(src, -1)).

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Side-by-side display of the original image and the 180-degree rotated image.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg"):
    print("[*] Running Experiment 11: 180-Degree Clockwise Rotation")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    h, w = orig_img.shape[:2]
    print(f"[-] Original Dimensions: {w} x {h}")

    # Step 2: Perform 180-degree rotation
    rotated_180 = cv2.rotate(orig_img, cv2.ROTATE_180)
    print(f"[-] 180-degree rotation completed. Resolution preserved: {w} x {h}")

    # Step 3: Side-by-side comparison
    comparison = create_comparison(orig_img, rotated_180, title1="Original Image", title2="180-Deg Rotated")

    # Step 4: Display output
    display_and_wait("Experiment 11 - 180-Degree Clockwise Rotation", comparison)

if __name__ == "__main__":
    run_experiment()
