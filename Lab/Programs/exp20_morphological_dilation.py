"""
========================================================================================
EXPERIMENT 20: MORPHOLOGICAL DILATION WITH STRUCTURING ELEMENTS
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement the morphological dilation technique using OpenCV and evaluate how
    foreground regions expand under Rectangular, Cross, and Elliptical structuring elements.

THEORY & MATHEMATICAL FOUNDATION:
    Morphological dilation is an operation that grows or thickens foreground objects in
    an image. The extent and geometric direction of thickening are governed by the shape
    and size of the Structuring Element (SE) B:
        A (+) B = { z in E | (B_reflected)_z intersected with A != empty_set }

    OpenCV allows generating custom geometries via cv2.getStructuringElement():
        1. cv2.MORPH_RECT    : Uniform rectangular expansion.
        2. cv2.MORPH_CROSS   : Axis-aligned cross expansion.
        3. cv2.MORPH_ELLIPSE : Isotropic circular/radial expansion.

    Practical Significance:
        - Repairing broken characters in document OCR.
        - Connecting fractured road segments in satellite images.
        - Filling interior pores in segmented cell nuclei.

INPUT:
    assets/images/shapes.jpg

OUTPUT:
    Multi-panel display comparing the original image with dilation outputs across
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
    print(f"[*] Running Experiment 20: Morphological Dilation with Structuring Elements {ksize}")

    # Step 1: Read input image
    orig_img = load_image(image_name)

    # Step 2: Generate different structuring elements
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)

    # Step 3: Apply dilation with each kernel
    dilated_rect = cv2.dilate(orig_img, kernel_rect, iterations=1)
    dilated_cross = cv2.dilate(orig_img, kernel_cross, iterations=1)
    dilated_ellipse = cv2.dilate(orig_img, kernel_ellipse, iterations=1)

    print("[-] Computed dilation across Rect, Cross, and Elliptical structuring elements.")

    # Step 4: Add descriptive labels
    def add_label(img, text):
        out = img.copy()
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out

    p1 = add_label(orig_img, "Original Image")
    p2 = add_label(dilated_rect, "Dilation (MORPH_RECT)")
    p3 = add_label(dilated_cross, "Dilation (MORPH_CROSS)")
    p4 = add_label(dilated_ellipse, "Dilation (MORPH_ELLIPSE)")

    grid = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])

    # Step 5: Display output
    display_and_wait("Experiment 20 - Morphological Dilation Technique", grid)

if __name__ == "__main__":
    run_experiment()
