"""
========================================================================================
EXPERIMENT 06: IMAGE EROSION USING ERODE FUNCTION
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform basic morphological erosion on an image using OpenCV's erode function
    and observe its effect on image boundaries and noise.

THEORY & MATHEMATICAL FOUNDATION:
    Erosion is a fundamental morphological operation based on set theory. In grayscale
    or binary morphology, erosion slides a structuring element (kernel) across the image:
        (A (-) B)(x, y) = min_{(i, j) in B} A(x + i, y + j)

    A pixel in the original image is retained (as 1 / high intensity) ONLY if all pixels
    under the structuring element are 1; otherwise, it is eroded (set to 0 / minimum intensity).
    
    Key Effects:
        - Shrinks the size of bright foreground objects.
        - Strips away small, isolated background noise and thin boundary projections.
        - Disconnects two weakly bridged foreground components.

INPUT:
    assets/images/shapes.jpg (or sample.jpg)

OUTPUT:
    Side-by-side display of the original image and the eroded output.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="shapes.jpg", kernel_size=(5, 5), iterations=1):
    print(f"[*] Running Experiment 06: Image Erosion ({kernel_size}, iterations={iterations})")

    # Step 1: Read input image
    img = load_image(image_name)

    # Step 2: Define structuring element (kernel)
    kernel = np.ones(kernel_size, dtype=np.uint8)

    # Step 3: Apply erosion
    eroded_img = cv2.erode(img, kernel, iterations=iterations)
    print("[-] Erosion operation completed successfully.")

    # Step 4: Side-by-side comparison
    comparison = create_comparison(img, eroded_img, title1="Original Image", title2=f"Eroded (iter={iterations})")

    # Step 5: Display result
    display_and_wait("Experiment 06 - Image Erosion", comparison)

if __name__ == "__main__":
    run_experiment()
