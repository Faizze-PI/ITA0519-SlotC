"""
========================================================================================
EXPERIMENT 02: IMAGE BLURRING USING GAUSSIAN BLUR
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform image smoothing by reading an image in Python and applying a
    Gaussian Blur filter using OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Gaussian Blur is a linear filtering technique used for smoothing, noise reduction,
    and anti-aliasing. Unlike simple box/mean blurring, Gaussian blurring uses a 2D
    Gaussian distribution kernel where center pixels have higher weight than boundary pixels:
        G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))

    Key Parameters:
        - ksize: The kernel size (width, height), which must be positive and odd (e.g., 5x5, 9x9).
        - sigmaX: Gaussian kernel standard deviation along the X-axis. Setting 0 computes sigma
          automatically from kernel size: sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8.

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Side-by-side display of the original image and the Gaussian-blurred image.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg", kernel_size=(11, 11), sigma_x=0):
    print(f"[*] Running Experiment 02: Gaussian Blur with kernel {kernel_size}")

    # Step 1: Read the input image
    orig_img = load_image(image_name)

    # Step 2: Apply Gaussian Blur
    # Ensure kernel dimensions are odd integers
    k_w, k_h = kernel_size
    if k_w % 2 == 0: k_w += 1
    if k_h % 2 == 0: k_h += 1
    
    blurred_img = cv2.GaussianBlur(orig_img, (k_w, k_h), sigmaX=sigma_x)
    print(f"[-] Gaussian Blur successfully applied with kernel ({k_w}, {k_h}) and sigmaX={sigma_x}")

    # Step 3: Create side-by-side comparison
    comparison = create_comparison(orig_img, blurred_img, title1="Original Sharp", title2=f"Gaussian Blur ({k_w}x{k_h})")

    # Step 4: Display output
    display_and_wait("Experiment 02 - Gaussian Blur", comparison)

if __name__ == "__main__":
    run_experiment()
