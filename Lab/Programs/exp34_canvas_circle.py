"""
========================================================================================
EXPERIMENT 34: SYNTHETIC CANVAS WITH CIRCLE SHAPE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to generate a white canvas whose dimensions are entered
    by the user, and draw a geometric circle shape using OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Circles are defined by a center coordinate point (c_x, c_y) and a radial distance r:
        (x - c_x)^2 + (y - c_y)^2 = r^2

    In OpenCV, cv2.circle() rasterizes circles efficiently:
        cv2.circle(img, center, radius, color, thickness, lineType)
    where:
        - center: (x, y) center coordinate tuple.
        - radius: Integer radius of the circle in pixels.
        - color: BGR color tuple, e.g. (0, 140, 255) for Orange.
        - thickness: Line width (or -1 / cv2.FILLED for solid fill).
        - lineType: Anti-aliasing flag (cv2.LINE_AA for smooth circular curves).

INPUT:
    User-specified canvas dimensions (e.g. Width=500, Height=400).

OUTPUT:
    Display window showing the white canvas containing the drawn circle.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import display_and_wait

def create_circle_canvas(width=500, height=400):
    print(f"[*] Running Experiment 34: Circle Shape on Canvas ({width}x{height})")

    # Step 1: Generate white canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Step 2: Compute center coordinate and radius
    center = (width // 2, height // 2)
    radius = min(width, height) // 3

    # Step 3: Draw Circle
    # Concentric outer circle outline
    cv2.circle(canvas, center, radius, (0, 140, 255), thickness=4, lineType=cv2.LINE_AA)
    # Inner filled circle
    cv2.circle(canvas, center, radius - 15, (220, 240, 255), thickness=-1, lineType=cv2.LINE_AA)
    # Center pinpoint
    cv2.circle(canvas, center, 5, (0, 0, 255), thickness=-1)

    # Step 4: Add descriptive annotations
    cv2.putText(canvas, f"Center: {center}, Radius: {radius}px", (center[0] - 120, center[1] + radius + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)

    # Step 5: Display result
    display_and_wait("Experiment 34 - Circle on Canvas", canvas)
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

    create_circle_canvas(width, height)

if __name__ == "__main__":
    run_experiment()
