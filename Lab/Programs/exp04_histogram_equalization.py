"""
========================================================================================
EXPERIMENT 04: HISTOGRAM EQUALIZATION AND COMPARISON
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement Histogram Equalization on a given image and compare the enhanced
    contrast output with the original image using OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Histogram Equalization is an intensity transformation technique that enhances
    image contrast by flattening and stretching the intensity distribution across the
    entire dynamic range [0 - 255]. It works by mapping pixel intensities through their
    normalized Cumulative Distribution Function (CDF):
        s_k = (L - 1) * sum_{j=0}^{k} p_r(r_j)
    where:
        - L is the total number of gray levels (typically 256).
        - p_r(r_j) is the probability of intensity level r_j occurring.

    For Grayscale:
        Applied directly via cv2.equalizeHist().
    For Color (BGR):
        Equalizing B, G, R channels independently distorts hue. The professional approach
        converts BGR -> YCrCb (or LAB), applies equalizeHist() to the luminance (Y)
        channel, and converts back to BGR.

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Side-by-side comparison of the original image and the contrast-equalized image.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg"):
    print("[*] Running Experiment 04: Histogram Equalization")

    # Step 1: Load original image
    bgr_img = load_image(image_name)

    # Step 2: Grayscale Equalization
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(gray_img)

    # Step 3: Color Luminance Equalization (Y channel in YCrCb space)
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    equalized_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    print("[-] Computed Histogram Equalization for both Grayscale and Color.")

    # Step 4: Side-by-side comparison for Grayscale
    gray_comparison = create_comparison(gray_img, equalized_gray, title1="Original Grayscale", title2="Equalized Grayscale")

    # Step 5: Side-by-side comparison for Color
    color_comparison = create_comparison(bgr_img, equalized_bgr, title1="Original BGR", title2="Equalized Luminance (Y)")

    # Display Grayscale comparison first
    display_and_wait("Exp 04 - Grayscale Equalization", gray_comparison)
    # Display Color comparison second
    display_and_wait("Exp 04 - Color Luminance Equalization", color_comparison)

if __name__ == "__main__":
    run_experiment()
