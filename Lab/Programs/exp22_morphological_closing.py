"""
========================================================================================
EXPERIMENT 22: MORPHOLOGICAL CLOSING TECHNIQUE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement the Morphological Closing operation using OpenCV and observe how it
    fills interior holes and bridges gaps within foreground objects.

THEORY & MATHEMATICAL FOUNDATION:
    Morphological Closing is defined as a Dilation followed by an Erosion using the
    same Structuring Element (SE) B:
        A * B = (A (+) B) (-) B

    Properties & Mechanism:
        1. Idempotent: (A * B) * B = A * B (repeating yields no additional change).
        2. Extensive: A is a subset of (A * B) (never shrinks original foreground).
        3. Hole-Filling: Small dark background holes or cracks completely covered by B
           are eliminated during dilation and cannot be carved back during erosion.
        4. Bridging: Merges close components and smooths narrow concave indentations.

INPUT:
    assets/images/shapes.jpg

OUTPUT:
    Side-by-side display of the original image with holes and the closed output.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="shapes.jpg", ksize=(15, 15)):
    print(f"[*] Running Experiment 22: Morphological Closing (Kernel={ksize})")

    # Step 1: Read input image
    orig_img = load_image(image_name)

    # Step 2: Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    # Step 3: Apply Morphological Closing
    closed_img = cv2.morphologyEx(orig_img, cv2.MORPH_CLOSE, kernel)
    print("[-] Closing completed (interior holes filled and gaps bridged).")

    # Step 4: Create side-by-side comparison
    comparison = create_comparison(orig_img, closed_img, title1="Original (with Holes/Cracks)", title2="Closed (MORPH_CLOSE)")

    # Step 5: Display result
    display_and_wait("Experiment 22 - Morphological Closing", comparison)

if __name__ == "__main__":
    run_experiment()
