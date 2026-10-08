"""
========================================================================================
EXPERIMENT 01: IMAGE READING AND CONVERSION TO GRAYSCALE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform basic image handling operations: read an input image using Python
    and convert the colored image into grayscale using OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    A color digital image is typically represented in BGR format with three color
    channels (Blue, Green, Red), each having 8-bit intensity values [0 - 255].
    Converting an image to grayscale simplifies algorithmic complexity, reduces memory
    by 66% (from 3 bytes to 1 byte per pixel), and eliminates sensitivity to color
    variations while preserving luminance and edge boundaries.

    OpenCV calculates luminance (Y) using the standard ITU-R Recommendation BT.601:
        Y = 0.299 * R + 0.587 * G + 0.114 * B
    (In BGR ordering: Y = 0.114 * B + 0.587 * G + 0.299 * R)

INPUT:
    assets/images/sample.jpg (or any valid color image)

OUTPUT:
    Side-by-side display of the original BGR image and the resulting Grayscale image.
========================================================================================
"""

import os
import sys
import cv2

# Import shared helper utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg"):
    print("[*] Running Experiment 01: Convert Image to Grayscale")

    # Step 1: Read the input color image
    color_img = load_image(image_name, cv2.IMREAD_COLOR)
    h, w, c = color_img.shape
    print(f"[-] Loaded original image: Resolution = {w}x{h}, Channels = {c}")

    # Step 2: Convert color space from BGR to Grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    print(f"[-] Grayscale image created: Resolution = {gray_img.shape[1]}x{gray_img.shape[0]}, Channels = 1")

    # Step 3: Create professional side-by-side comparison
    comparison_view = create_comparison(color_img, gray_img, title1="Original BGR", title2="Grayscale")

    # Step 4: Display output window
    display_and_wait("Experiment 01 - Grayscale Conversion", comparison_view)

if __name__ == "__main__":
    run_experiment()
