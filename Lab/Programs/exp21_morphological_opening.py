"""
========================================================================================
EXPERIMENT 21: MORPHOLOGICAL OPENING TECHNIQUE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement the Morphological Opening operation using OpenCV and observe how it
    eliminates foreground noise while preserving overall object geometry.

THEORY & MATHEMATICAL FOUNDATION:
    Morphological Opening is defined as an Erosion followed by a Dilation using the
    same Structuring Element (SE) B:
        A o B = (A (-) B) (+) B

    Properties & Mechanism:
        1. Idempotent: (A o B) o B = A o B (repeating the operation causes no further change).
        2. Anti-extensive: A o B is a subset of A (foreground does not expand).
        3. Elimination of Small Objects: Any bright feature that cannot completely contain
           the structuring element B is erased during the erosion step and not restored
           by the subsequent dilation.
        4. Smoothing: Smooths convex contours and breaks thin connecting isthmuses.

INPUT:
    assets/images/shapes.jpg

OUTPUT:
    Side-by-side display of the original noisy image and the opened (noise-removed) output.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="shapes.jpg", ksize=(9, 9)):
    print(f"[*] Running Experiment 21: Morphological Opening (Kernel={ksize})")

    # Step 1: Read input image
    orig_img = load_image(image_name)

    # Step 2: Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    # Step 3: Apply Morphological Opening
    opened_img = cv2.morphologyEx(orig_img, cv2.MORPH_OPEN, kernel)
    print("[-] Opening completed (isolated small noise dots eliminated).")

    # Step 4: Create side-by-side comparison
    comparison = create_comparison(orig_img, opened_img, title1="Original (with Noise Dots)", title2="Opened (MORPH_OPEN)")

    # Step 5: Display result
    display_and_wait("Experiment 21 - Morphological Opening", comparison)

if __name__ == "__main__":
    run_experiment()
