"""
========================================================================================
EXPERIMENT 31: IMAGE SEGMENTATION VIA THRESHOLDING
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement an image segmentation algorithm using OpenCV to segment an input
    image into foreground and background regions based on given threshold values.

THEORY & MATHEMATICAL FOUNDATION:
    Threshold-based segmentation converts a grayscale image into a binary mask by
    comparing each pixel intensity f(x, y) against a threshold value T:
        Binary Thresholding:
            g(x, y) = maxVal    if f(x, y) >= T
                      0         otherwise

        Binary Inverted Thresholding:
            g(x, y) = 0         if f(x, y) >= T
                      maxVal    otherwise

    Otsu's Thresholding (Automatic Optimal Threshold):
        Nobuyuki Otsu's method (1979) automatically calculates the optimal threshold T*
        by maximizing the between-class variance sigma_B^2(T) of foreground and background:
            sigma_B^2(T) = w0(T) * w1(T) * [ mu0(T) - mu1(T) ]^2

    OpenCV API:
        ret, thresh = cv2.threshold(src, thresh, maxval, type)

INPUT:
    assets/images/sample.jpg (or shapes.jpg)

OUTPUT:
    Multi-panel display showing: Original Grayscale, Binary Threshold, Inverted Threshold,
    and Otsu's optimal segmentation.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, display_and_wait

def run_experiment(image_name="sample.jpg", threshold_val=127):
    print(f"[*] Running Experiment 31: Threshold Segmentation (Manual T={threshold_val})")

    # Step 1: Read input and convert to grayscale
    orig_img = load_image(image_name)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Step 2: Binary Thresholding
    _, thresh_binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    # Step 3: Binary Inverted Thresholding
    _, thresh_inv = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # Step 4: Otsu's Automatic Thresholding
    otsu_val, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"[-] Manual Threshold = {threshold_val}, Otsu Optimal Threshold = {otsu_val:.1f}")

    # Step 5: Format multi-panel display
    def label_panel(img, text):
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return out

    p1 = label_panel(gray, "Original Grayscale")
    p2 = label_panel(thresh_binary, f"Binary Threshold (T={threshold_val})")
    p3 = label_panel(thresh_inv, f"Inverted Binary (T={threshold_val})")
    p4 = label_panel(thresh_otsu, f"Otsu Optimal (T*={otsu_val:.1f})")

    grid = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])

    # Step 6: Display result
    display_and_wait("Experiment 31 - Threshold Segmentation", grid)

if __name__ == "__main__":
    run_experiment()
