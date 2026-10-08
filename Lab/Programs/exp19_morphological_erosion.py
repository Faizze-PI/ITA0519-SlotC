"""
========================================================================================
EXPERIMENT 19: MORPHOLOGICAL EROSION WITH STRUCTURING ELEMENTS
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement the morphological erosion technique using OpenCV and evaluate the
    effects of different structuring element shapes (Rectangle, Cross, Ellipse).

THEORY & MATHEMATICAL FOUNDATION:
    Morphological erosion shrinks the foreground (typically white/bright pixels) by
    checking the containment of a Structuring Element (SE) kernel B within set A:
        A (-) B = { z in E | B_z subset of A }

    In OpenCV, structuring elements of distinct geometries can be synthesized using:
        cv2.getStructuringElement(shape, ksize)
    Shapes:
        1. cv2.MORPH_RECT    : Standard rectangular box (erodes diagonally and orthogonally).
        2. cv2.MORPH_CROSS   : Cross (+) shaped kernel (erodes primarily along axes).
        3. cv2.MORPH_ELLIPSE : Discrete ellipse (erodes isotropically / rounded boundaries).

INPUT:
    assets/images/shapes.jpg

OUTPUT:
    Multi-panel display comparing the original image with erosion outputs across
    Rectangular, Cross, and Elliptical structuring elements.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, display_and_wait

def run_experiment(image_name="shapes.jpg", ksize=(7, 7)):
    print(f"[*] Running Experiment 19: Morphological Erosion with Structuring Elements {ksize}")

    # Step 1: Read input image
    orig_img = load_image(image_name)

    # Step 2: Generate different structuring elements
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)

    # Step 3: Apply erosion with each kernel
    eroded_rect = cv2.erode(orig_img, kernel_rect, iterations=1)
    eroded_cross = cv2.erode(orig_img, kernel_cross, iterations=1)
    eroded_ellipse = cv2.erode(orig_img, kernel_ellipse, iterations=1)

    print("[-] Computed erosion across Rect, Cross, and Elliptical structuring elements.")

    # Step 4: Add descriptive labels
    def add_label(img, text):
        out = img.copy()
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out

    p1 = add_label(orig_img, "Original Image")
    p2 = add_label(eroded_rect, "Erosion (MORPH_RECT)")
    p3 = add_label(eroded_cross, "Erosion (MORPH_CROSS)")
    p4 = add_label(eroded_ellipse, "Erosion (MORPH_ELLIPSE)")

    grid = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])

    # Step 5: Display output
    display_and_wait("Experiment 19 - Morphological Erosion Technique", grid)

if __name__ == "__main__":
    run_experiment()
