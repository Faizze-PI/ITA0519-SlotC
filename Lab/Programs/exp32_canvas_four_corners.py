"""
========================================================================================
EXPERIMENT 32: SYNTHETIC CANVAS WITH FOUR COLORED CORNER BOXES
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to generate a white canvas whose dimensions are entered
    by the user, and construct four colored boxes (Black, Blue, Green, Red) on each corner,
    with each box occupying exactly 1/10th of the image dimensions.

THEORY & MATHEMATICAL FOUNDATION:
    In computer vision, digital canvases are represented as multi-dimensional NumPy arrays.
    A pure white 3-channel BGR image of size (Height x Width) is generated using:
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    The bounding dimensions for each corner box are computed as:
        box_h = height // 10
        box_w = width // 10

    Corner Slicing & BGR Assignments:
    1. Top-Left     : Black [0, 0, 0]     -> canvas[0 : box_h, 0 : box_w]
    2. Top-Right    : Blue  [255, 0, 0]   -> canvas[0 : box_h, width - box_w : width]
    3. Bottom-Left  : Green [0, 255, 0]   -> canvas[height - box_h : height, 0 : box_w]
    4. Bottom-Right : Red   [0, 0, 255]   -> canvas[height - box_h : height, width - box_w : width]

INPUT:
    User-specified canvas dimensions (e.g. Width=500, Height=400).

OUTPUT:
    Display window showing the white canvas with the 4 distinct colored corner boxes.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import display_and_wait

def create_four_corners_canvas(width=500, height=400):
    print(f"[*] Running Experiment 32: Synthetic White Canvas ({width}x{height}) with 4 Corner Boxes")

    # Step 1: Create 3D white image canvas (all 255s)
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Step 2: Compute 1/10th box dimensions
    box_h = max(1, height // 10)
    box_w = max(1, width // 10)

    print(f"[-] Canvas Size = {width}x{height}, Corner Box Size (1/10th) = {box_w}x{box_h}")

    # Step 3: Draw corner boxes in BGR format
    # 1. Top-Left: Black
    canvas[0:box_h, 0:box_w] = [0, 0, 0]
    
    # 2. Top-Right: Blue
    canvas[0:box_h, width - box_w:width] = [255, 0, 0]
    
    # 3. Bottom-Left: Green
    canvas[height - box_h:height, 0:box_w] = [0, 255, 0]
    
    # 4. Bottom-Right: Red
    canvas[height - box_h:height, width - box_w:width] = [0, 0, 255]

    # Step 4: Add descriptive center text
    cv2.putText(canvas, f"Canvas: {width}x{height}", (width // 2 - 100, height // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)
    cv2.putText(canvas, f"Corner Box (1/10): {box_w}x{box_h}", (width // 2 - 120, height // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

    # Step 5: Display result
    display_and_wait("Experiment 32 - 4 Corner Boxes Canvas", canvas)
    return canvas

def run_experiment():
    # Supports optional user interactive input or clean defaults
    try:
        if sys.stdin.isatty():
            w_input = input("Enter canvas width [Default 500]: ").strip()
            h_input = input("Enter canvas height [Default 400]: ").strip()
            width = int(w_input) if w_input else 500
            height = int(h_input) if h_input else 400
        else:
            width, height = 500, 400
    except Exception:
        width, height = 500, 400

    create_four_corners_canvas(width, height)

if __name__ == "__main__":
    run_experiment()
