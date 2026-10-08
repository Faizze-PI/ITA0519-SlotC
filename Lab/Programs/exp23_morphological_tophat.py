"""
========================================================================================
EXPERIMENT 23: TOP HAT MORPHOLOGICAL OPERATION
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement the Top Hat morphological operation using OpenCV and evaluate its
    effectiveness in isolating elements brighter than their surrounding neighborhood.

THEORY & MATHEMATICAL FOUNDATION:
    The Top Hat transform (also called White Top Hat) is defined as the algebraic
    difference between the original image and its morphological opening:
        TopHat(A) = A - (A o B)
    where (A o B) is the opening of image A by structuring element B.

    Mechanism:
        Since the opening operation removes bright features smaller than the structuring
        element B while leaving large background structures intact, subtracting the opened
        image from the original isolates precisely those small, bright foreground details.

    Applications:
        - Correcting non-uniform background illumination.
        - Extracting bright text or micro-features from unevenly lit surfaces.
        - Detecting stars or celestial bodies against varying atmospheric light.

INPUT:
    assets/images/shapes.jpg (or sample.jpg)

OUTPUT:
    Side-by-side display of the original image and the extracted Top Hat features.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="shapes.jpg", ksize=(15, 15)):
    print(f"[*] Running Experiment 23: Top Hat Morphological Operation (Kernel={ksize})")

    # Step 1: Read input image
    orig_img = load_image(image_name)

    # Step 2: Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    # Step 3: Compute Top Hat
    tophat_img = cv2.morphologyEx(orig_img, cv2.MORPH_TOPHAT, kernel)
    print("[-] Top Hat transform calculated (bright elements isolated).")

    # Step 4: Create side-by-side comparison
    comparison = create_comparison(orig_img, tophat_img, title1="Original Image", title2="Top Hat (Bright Elements)")

    # Step 5: Display result
    display_and_wait("Experiment 23 - Morphological Top Hat", comparison)

if __name__ == "__main__":
    run_experiment()
