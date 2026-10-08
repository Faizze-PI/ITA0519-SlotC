"""
========================================================================================
EXPERIMENT 16: SOBEL ALGORITHM FOR IMAGE GRADIENT FILTERING
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement the Sobel gradient filtering algorithm using OpenCV to detect horizontal,
    vertical, and combined edge features in an input image.

THEORY & MATHEMATICAL FOUNDATION:
    The Sobel-Feldman operator is a discrete differentiation operator computing an
    approximation of the gradient of image intensity function. It combines Gaussian
    smoothing with differentiation to reduce high-frequency noise sensitivity.
    
    It convolves the image with two orthogonal 3x3 convolution kernels:
        Horizontal Sobel (Gx, detects vertical edges):
            [ -1  0  +1 ]
            [ -2  0  +2 ]
            [ -1  0  +1 ]

        Vertical Sobel (Gy, detects horizontal edges):
            [ -1  -2  -1 ]
            [  0   0   0 ]
            [ +1  +2  +1 ]

    Gradient Magnitude:
        G = sqrt(Gx^2 + Gy^2)   ~=   |Gx| + |Gy|

    In OpenCV, computing gradients into CV_64F prevents negative underflow, after which
    cv2.convertScaleAbs() converts the signed gradient back into displayable uint8.

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Multi-panel display showing: Original Grayscale, Sobel X, Sobel Y, and Combined Sobel.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, display_and_wait

def run_experiment(image_name="sample.jpg", ksize=3):
    print(f"[*] Running Experiment 16: Sobel Filter (kernel size={ksize})")

    # Step 1: Read input image and convert to grayscale
    orig_img = load_image(image_name)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Step 2: Compute horizontal gradient (Gx, detects vertical edges)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)

    # Step 3: Compute vertical gradient (Gy, detects horizontal edges)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # Step 4: Combine gradients using weighted sum approximation
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    print("[-] Sobel gradients (X, Y, Combined) computed.")

    # Step 5: Format a 2x2 grid comparison
    top_row = np.hstack([gray, abs_sobel_x])
    bottom_row = np.hstack([abs_sobel_y, sobel_combined])
    grid = np.vstack([top_row, bottom_row])

    # Convert to 3 channels for drawing colored annotations
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    cv2.putText(grid_bgr, "Original Grayscale", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(grid_bgr, "Sobel X (Vertical Edges)", (w + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(grid_bgr, "Sobel Y (Horizontal Edges)", (15, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(grid_bgr, "Sobel Combined (Total Gradient)", (w + 15, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Step 6: Display result
    display_and_wait("Experiment 16 - Sobel Gradient Filter", grid_bgr)

if __name__ == "__main__":
    run_experiment()
