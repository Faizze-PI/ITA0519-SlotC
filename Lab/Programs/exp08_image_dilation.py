"""
========================================================================================
EXPERIMENT 08: IMAGE DILATION USING DILATE FUNCTION
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform basic morphological dilation on an image using OpenCV's dilate function
    and observe how foreground regions expand and holes fill in.

THEORY & MATHEMATICAL FOUNDATION:
    Dilation is the dual operation of erosion in mathematical morphology. It convolves
    a structuring element (kernel) B over an image A, computing the local maximum:
        (A (+) B)(x, y) = max_{(i, j) in B} A(x - i, y - j)

    A pixel in the output is set to 1 if at least one pixel under the kernel is 1.
    
    Key Effects:
        - Expands the boundaries of foreground objects.
        - Fills in small dark holes and cracks within objects.
        - Bridges small gaps between adjacent foreground shapes.

INPUT:
    assets/images/shapes.jpg (or sample.jpg)

OUTPUT:
    Side-by-side display of the original image and the dilated output.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="shapes.jpg", kernel_size=(5, 5), iterations=1):
    print(f"[*] Running Experiment 08: Image Dilation ({kernel_size}, iterations={iterations})")

    # Step 1: Read input image
    img = load_image(image_name)

    # Step 2: Define structuring element (kernel)
    kernel = np.ones(kernel_size, dtype=np.uint8)

    # Step 3: Apply dilation
    dilated_img = cv2.dilate(img, kernel, iterations=iterations)
    print("[-] Dilation operation completed successfully.")

    # Step 4: Side-by-side comparison
    comparison = create_comparison(img, dilated_img, title1="Original Image", title2=f"Dilated (iter={iterations})")

    # Step 5: Display result
    display_and_wait("Experiment 08 - Image Dilation", comparison)

if __name__ == "__main__":
    run_experiment()
