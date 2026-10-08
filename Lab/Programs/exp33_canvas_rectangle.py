"""
========================================================================================
EXPERIMENT 33: SYNTHETIC CANVAS WITH RECTANGLE SHAPE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to generate a white canvas whose dimensions are entered
    by the user, and draw a geometric rectangle shape using OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Drawing 2D primitive geometries is an essential visualization and data annotation
    skill in computer vision.
    
    A white canvas is generated using NumPy:
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    The rectangle is drawn via cv2.rectangle():
        cv2.rectangle(img, pt1, pt2, color, thickness)
    where:
        - pt1: Coordinate tuple of the top-left vertex (x1, y1).
        - pt2: Coordinate tuple of the bottom-right vertex (x2, y2).
        - color: BGR color tuple, e.g. (0, 0, 255) for Red.
        - thickness: Line thickness in pixels (or -1 / cv2.FILLED for a solid filled shape).

INPUT:
    User-specified canvas dimensions (e.g. Width=500, Height=400).

OUTPUT:
    Display window showing the white canvas containing the drawn rectangle.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import display_and_wait

def create_rectangle_canvas(width=500, height=400):
    print(f"[*] Running Experiment 33: Rectangle Shape on Canvas ({width}x{height})")

    # Step 1: Generate white canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Step 2: Define centered rectangle coordinates
    margin_x = int(width * 0.2)
    margin_y = int(height * 0.2)
    pt1 = (margin_x, margin_y)
    pt2 = (width - margin_x, height - margin_y)

    # Step 3: Draw Rectangle
    # Border outline in Navy Blue
    cv2.rectangle(canvas, pt1, pt2, (180, 50, 20), thickness=4)
    # Interior filled accent in light blue
    cv2.rectangle(canvas, (pt1[0] + 15, pt1[1] + 15), (pt2[0] - 15, pt2[1] - 15), (240, 210, 180), thickness=-1)

    # Step 4: Add descriptive annotations
    cv2.putText(canvas, f"Rectangle: pt1={pt1}, pt2={pt2}", (margin_x, margin_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

    # Step 5: Display result
    display_and_wait("Experiment 33 - Rectangle on Canvas", canvas)
    return canvas

def run_experiment():
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

    create_rectangle_canvas(width, height)

if __name__ == "__main__":
    run_experiment()
