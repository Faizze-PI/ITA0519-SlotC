"""
========================================================================================
EXPERIMENT 13: AFFINE TRANSFORMATION
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform an Affine Transformation on an input image using Python and OpenCV
    by calculating a 2x3 transformation matrix from three coordinate point correspondences.

THEORY & MATHEMATICAL FOUNDATION:
    An Affine Transformation is any geometric transformation that preserves collinearity
    (all points lying on a line continue to lie on a line) and ratios of distances along
    parallel lines. Parallel lines before the transformation remain parallel after it.
    
    The transformation relates coordinates (x, y) to (x', y') using a 2x3 matrix M:
        [ x' ]   [ a11  a12  b1 ] [ x ]
        [ y' ] = [ a21  a22  b2 ] [ y ]
                                  [ 1 ]
    Because there are 6 degrees of freedom (a11, a12, a21, a22, b1, b2), exactly 3
    non-collinear pairs of corresponding points are required to uniquely solve the system:
        M = cv2.getAffineTransform(pts_src, pts_dst)
        output = cv2.warpAffine(src, M, (width, height))

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Side-by-side display of the original image and the affine-transformed image.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg"):
    print("[*] Running Experiment 13: Affine Transformation")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    rows, cols = orig_img.shape[:2]

    # Step 2: Define 3 source points and 3 corresponding destination points
    # Three triangle vertices in the source image
    pts_src = np.float32([
        [50, 50],
        [200, 50],
        [50, 200]
    ])

    # Corresponding mapped target positions (simulating shear, shift and rotation)
    pts_dst = np.float32([
        [70, 100],
        [220, 50],
        [150, 250]
    ])

    # Step 3: Compute the 2x3 Affine Transformation Matrix
    M = cv2.getAffineTransform(pts_src, pts_dst)
    print("[-] Computed 2x3 Affine Matrix M:")
    print(M)

    # Step 4: Apply affine warp to the image
    affine_warped = cv2.warpAffine(orig_img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))

    # Step 5: Draw visual anchor dots on the source points for demonstration
    annotated_orig = orig_img.copy()
    for pt in pts_src.astype(int):
        cv2.circle(annotated_orig, tuple(pt), 6, (0, 0, 255), -1)

    annotated_warped = affine_warped.copy()
    for pt in pts_dst.astype(int):
        cv2.circle(annotated_warped, tuple(pt), 6, (0, 255, 0), -1)

    # Step 6: Create comparison view
    comparison = create_comparison(annotated_orig, annotated_warped, title1="Original (with 3 Anchor Pts)", title2="Affine Warped")

    # Step 7: Display result
    display_and_wait("Experiment 13 - Affine Transformation", comparison)

if __name__ == "__main__":
    run_experiment()
