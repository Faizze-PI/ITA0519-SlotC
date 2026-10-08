"""
========================================================================================
EXPERIMENT 12: 270-DEGREE ROTATION CLOCKWISE (90-DEG COUNTER-CLOCKWISE)
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform a 270-degree clockwise rotation (equivalent to 90 degrees counter-clockwise)
    on a given image using Python and OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Rotating an image 270 degrees clockwise is mathematically identical to a 90-degree
    counter-clockwise rotation. For an image of dimensions (W x H), each pixel (x, y)
    maps to:
        x' = y
        y' = W - 1 - x
    The resulting image dimensions swap from (W x H) to (H x W).

    OpenCV provides an optimized function:
        cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Side-by-side display of the original image and the 270-degree rotated image.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg"):
    print("[*] Running Experiment 12: 270-Degree Clockwise Rotation")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    h, w = orig_img.shape[:2]
    print(f"[-] Original Dimensions: {w} x {h}")

    # Step 2: Perform 270-degree clockwise rotation (90 counter-clockwise)
    rotated_270 = cv2.rotate(orig_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    new_h, new_w = rotated_270.shape[:2]
    print(f"[-] Rotated Dimensions: {new_w} x {new_h}")

    # Step 3: Side-by-side comparison
    comparison = create_comparison(orig_img, rotated_270, title1="Original Image", title2="270-Deg Clockwise")

    # Step 4: Display output
    display_and_wait("Experiment 12 - 270-Degree Clockwise Rotation", comparison)

if __name__ == "__main__":
    run_experiment()
