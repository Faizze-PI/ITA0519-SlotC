"""
========================================================================================
EXPERIMENT 03: OUTLINE EXTRACTION USING CANNY EDGE DETECTION
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To detect and display the edge boundaries and structural outlines of an image
    using the Canny edge detection algorithm in OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Developed by John F. Canny in 1986, the Canny edge detector is an optimal multi-stage
    edge detection operator consisting of four sequential steps:
    1. Gaussian Filtering: Convolves image with a Gaussian kernel to suppress noise.
    2. Intensity Gradient Calculation: Uses Sobel kernels (Gx, Gy) to calculate gradient
       magnitude G = sqrt(Gx^2 + Gy^2) and direction theta = arctan(Gy / Gx).
    3. Non-Maximum Suppression (NMS): Scans across gradient directions to suppress pixels
       that are not local maxima, producing thin, 1-pixel wide contours.
    4. Hysteresis Thresholding: Uses two thresholds (T_low, T_high):
       - Pixels with gradient > T_high are declared definite edges.
       - Pixels with gradient < T_low are rejected.
       - Pixels between T_low and T_high are retained only if connected to a definite edge.

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Side-by-side display of the input image and the binary edge outline.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg", threshold1=100, threshold2=200):
    print(f"[*] Running Experiment 03: Canny Outline Detection (T_low={threshold1}, T_high={threshold2})")

    # Step 1: Read input image
    orig_img = load_image(image_name)

    # Step 2: Convert to grayscale for gradient analysis
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Step 3: Compute Canny edge outline
    edges = cv2.Canny(gray_img, threshold1=threshold1, threshold2=threshold2)
    print("[-] Canny edge extraction complete.")

    # Step 4: Side-by-side visual comparison
    comparison = create_comparison(orig_img, edges, title1="Original Image", title2="Canny Outline")

    # Step 5: Display result
    display_and_wait("Experiment 03 - Canny Edge Outline", comparison)

if __name__ == "__main__":
    run_experiment()
