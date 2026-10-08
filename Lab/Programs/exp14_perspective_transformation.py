"""
========================================================================================
EXPERIMENT 14: PERSPECTIVE TRANSFORMATION (HOMOGRAPHY)
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform Perspective Transformation on an input image using Python and OpenCV
    by computing a 3x3 homography matrix from four corresponding coordinate points.

THEORY & MATHEMATICAL FOUNDATION:
    A Perspective Transformation (projective transformation or homography) projects
    points from one 3D viewing plane onto another. While straight lines remain straight,
    parallel lines generally converge toward vanishing points.
    
    It is governed by a 3x3 matrix H with 8 degrees of freedom:
        [ x' ]   [ h11  h12  h13 ] [ x ]
        [ y' ] ~ [ h21  h22  h23 ] [ y ]
        [ 1  ]   [ h31  h32  h33 ] [ 1 ]

    Because there are 8 unknowns (h33 is normalized to 1), exactly 4 non-collinear
    point correspondences are required:
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        output = cv2.warpPerspective(src, M, (width, height))

    Common Real-World Applications:
        - Document scanner apps (keystone correction / bird's-eye rectification).
        - Autonomous driving lane detection (inverse perspective mapping).

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Side-by-side display of the original image (with 4 quad corners) and the rectified
    perspective output.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg"):
    print("[*] Running Experiment 14: Perspective Transformation")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    h, w = orig_img.shape[:2]

    # Step 2: Define 4 quadrilateral source points (simulating a tilted book/screen)
    pts_src = np.float32([
        [int(w * 0.15), int(h * 0.20)],  # Top-Left
        [int(w * 0.85), int(h * 0.10)],  # Top-Right
        [int(w * 0.05), int(h * 0.85)],  # Bottom-Left
        [int(w * 0.95), int(h * 0.90)]   # Bottom-Right
    ])

    # Step 3: Define 4 destination points (rectified orthogonal rectangle)
    out_w, out_h = 450, 350
    pts_dst = np.float32([
        [0, 0],
        [out_w - 1, 0],
        [0, out_h - 1],
        [out_w - 1, out_h - 1]
    ])

    # Step 4: Compute 3x3 Perspective Transformation Matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    print("[-] Computed 3x3 Perspective Matrix M:")
    print(M)

    # Step 5: Warp perspective
    perspective_out = cv2.warpPerspective(orig_img, M, (out_w, out_h))

    # Step 6: Annotate source image with the quadrilateral boundary
    annotated_orig = orig_img.copy()
    cv2.polylines(annotated_orig, [pts_src.astype(int)], isClosed=True, color=(0, 255, 255), thickness=2)
    for idx, pt in enumerate(pts_src.astype(int)):
        cv2.circle(annotated_orig, tuple(pt), 6, (0, 0, 255), -1)
        cv2.putText(annotated_orig, f"P{idx+1}", (pt[0] + 8, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Step 7: Create comparison view
    comparison = create_comparison(annotated_orig, perspective_out, title1="Tilted Quadrilateral Region", title2="Bird's-Eye Perspective")

    # Step 8: Display result
    display_and_wait("Experiment 14 - Perspective Transformation", comparison)

if __name__ == "__main__":
    run_experiment()
