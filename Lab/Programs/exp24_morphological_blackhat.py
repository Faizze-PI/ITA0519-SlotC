"""
========================================================================================
EXPERIMENT 24: BLACK HAT MORPHOLOGICAL OPERATION
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement the Black Hat morphological operation using OpenCV and evaluate its
    effectiveness in isolating features darker than their surrounding background.

THEORY & MATHEMATICAL FOUNDATION:
    The Black Hat transform (also known as Bottom Hat) is defined as the algebraic
    difference between the morphological closing of the image and the input image itself:
        BlackHat(A) = (A * B) - A
    where (A * B) is the closing of image A by structuring element B.

    Mechanism:
        Since the closing operation fills in dark holes, gaps, and indentations smaller
        than the structuring element B, subtracting the original image from the closed
        image extracts precisely those dark details that were filled.

    Applications:
        - Extracting dark text or barcodes printed on uneven, bright surfaces.
        - Detecting dark cracks and fractures in industrial quality inspection.
        - Isolating dark retinal blood vessels in fundus imaging.

INPUT:
    assets/images/shapes.jpg

OUTPUT:
    Side-by-side display of the original image and the extracted Black Hat features.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="shapes.jpg", ksize=(15, 15)):
    print(f"[*] Running Experiment 24: Black Hat Morphological Operation (Kernel={ksize})")

    # Step 1: Read input image
    orig_img = load_image(image_name)

    # Step 2: Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    # Step 3: Compute Black Hat
    blackhat_img = cv2.morphologyEx(orig_img, cv2.MORPH_BLACKHAT, kernel)
    print("[-] Black Hat transform calculated (dark elements isolated).")

    # Step 4: Create side-by-side comparison
    comparison = create_comparison(orig_img, blackhat_img, title1="Original Image", title2="Black Hat (Dark Elements)")

    # Step 5: Display result
    display_and_wait("Experiment 24 - Morphological Black Hat", comparison)

if __name__ == "__main__":
    run_experiment()
