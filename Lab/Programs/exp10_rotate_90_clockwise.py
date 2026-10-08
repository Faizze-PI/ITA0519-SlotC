"""
========================================================================================
EXPERIMENT 10: 90-DEGREE ROTATION CLOCKWISE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform a 90-degree clockwise rotation on a given image using Python and OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    In 2D digital image processing, a clockwise rotation by 90 degrees transforms a
    pixel coordinate (x, y) in an image of size (W x H) to:
        x' = H - 1 - y
        y' = x
    The resulting image dimensions swap from (W x H) to (H x W).
    
    OpenCV provides an optimized hardware-accelerated function:
        cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)

    Academic Note on "along the y-axis":
        In 2D planar vision, standard rotation occurs around the z-axis (normal to the screen).
        If an instructor asks about a 3D rotation along the vertical y-axis, that corresponds
        to horizontal reflection/flipping (cv2.flip(src, 1)). This program demonstrates the
        standard 90-degree clockwise rotation as required by the syllabus, and also notes the
        y-axis reflection for viva clarification.

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Side-by-side display of the original image and the 90-degree clockwise rotated image.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg"):
    print("[*] Running Experiment 10: 90-Degree Clockwise Rotation")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    h, w = orig_img.shape[:2]
    print(f"[-] Original Dimensions: {w} x {h}")

    # Step 2: Perform 90-degree clockwise rotation
    rotated_90 = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)
    new_h, new_w = rotated_90.shape[:2]
    print(f"[-] Rotated Dimensions: {new_w} x {new_h}")

    # Step 3: Side-by-side comparison
    comparison = create_comparison(orig_img, rotated_90, title1="Original Image", title2="90-Deg Clockwise")

    # Step 4: Display output
    display_and_wait("Experiment 10 - 90-Degree Clockwise Rotation", comparison)

if __name__ == "__main__":
    run_experiment()
