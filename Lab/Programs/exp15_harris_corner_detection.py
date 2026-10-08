"""
========================================================================================
EXPERIMENT 15: HARRIS CORNER DETECTION
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To detect and highlight corner feature points in an input image using the
    Harris Corner Detection algorithm in OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    A corner is a point where two distinct edge directions intersect, causing high
    intensity gradients in all directions. The Harris Corner Detector evaluates the
    local autocorrelation matrix (structure tensor) M over a local window W:
        M = sum_{(x, y) in W} [ Ix^2     Ix*Iy ]
                              [ Ix*Iy   Iy^2   ]

    The corner response measure R is defined without explicit eigenvalue computation as:
        R = det(M) - k * (trace(M))^2
          = (lambda1 * lambda2) - k * (lambda1 + lambda2)^2
    where k is an empirical sensitivity parameter (typically 0.04 to 0.06).

    Classification:
        - |R| is small  -> Flat region (no edges or corners).
        - R < 0         -> Edge region (one dominant gradient direction).
        - R > threshold -> Corner region (both eigenvalues large).

    OpenCV Function:
        cv2.cornerHarris(src_gray, blockSize, ksize, k)

INPUT:
    assets/images/sample.jpg (or shapes.jpg)

OUTPUT:
    Side-by-side display of the input image and the image annotated with red corner markers.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg", block_size=2, ksize=3, k=0.04):
    print(f"[*] Running Experiment 15: Harris Corner Detection (blockSize={block_size}, ksize={ksize}, k={k})")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # Step 2: Convert to float32 as required by cv2.cornerHarris
    gray_float = np.float32(gray)

    # Step 3: Compute Harris Corner Response Map
    dst = cv2.cornerHarris(gray_float, blockSize=block_size, ksize=ksize, k=k)

    # Step 4: Dilate response map to make detected corner markers prominent
    dst_dilated = cv2.dilate(dst, None)

    # Step 5: Threshold for strong corners (e.g., top 1% of maximum response)
    threshold = 0.01 * dst_dilated.max()
    corner_count = int(np.sum(dst > threshold))
    print(f"[-] Total strong corner keypoints detected: {corner_count}")

    # Step 6: Mark detected corners in bright red (BGR: [0, 0, 255])
    annotated_img = orig_img.copy()
    annotated_img[dst > threshold] = [0, 0, 255]

    # Step 7: Create comparison view
    comparison = create_comparison(orig_img, annotated_img, title1="Original Image", title2=f"Harris Corners (Count: {corner_count})")

    # Step 8: Display result
    display_and_wait("Experiment 15 - Harris Corner Detection", comparison)

if __name__ == "__main__":
    run_experiment()
