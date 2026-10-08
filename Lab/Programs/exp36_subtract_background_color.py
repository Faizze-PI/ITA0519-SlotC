"""
========================================================================================
EXPERIMENT 36: BACKGROUND SUBTRACTION BASED ON COLOR LEVELS
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to subtract/remove the background of an input image
    based on specific color levels in the HSV color space using OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Color-based background subtraction (chroma keying) segments image pixels according
    to their chromatic signatures. In standard BGR color space, luminance variations
    complicate thresholding; converting to HSV (Hue, Saturation, Value) decouples color
    wavelength (Hue: [0 - 179]) from brightness (Value: [0 - 255]).

    Algorithmic Steps:
    1. Color Conversion: cv2.cvtColor(img, cv2.COLOR_BGR2HSV).
    2. Background Color Masking:
       Identify the HSV bounds for the dominant background hue:
           bg_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    3. Foreground Inversion:
       Invert the background mask to retain foreground pixels:
           fg_mask = cv2.bitwise_not(bg_mask)
    4. Bitwise Extraction:
       Apply bitwise AND with the original image:
           result = cv2.bitwise_and(img, img, mask=fg_mask)

INPUT:
    assets/images/color_object.jpg

OUTPUT:
    Side-by-side comparison of the original image and the image with background removed.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def subtract_background_by_color(image_name="color_object.jpg"):
    print("[*] Running Experiment 36: Subtract Background Based on Color Levels")

    # Step 1: Read input image
    orig_img = load_image(image_name)

    # Step 2: Convert to HSV color space
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    # Step 3: Define HSV range for background color
    # In color_object.jpg, background is cyan/light blue (Hue ~ 90-125)
    lower_bg = np.array([85, 50, 50])
    upper_bg = np.array([130, 255, 255])

    # Generate background mask
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)

    # Step 4: Invert mask to isolate foreground objects (Green circle & Red rectangle)
    fg_mask = cv2.bitwise_not(bg_mask)

    # Step 5: Clean mask using morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Step 6: Extract foreground on black canvas
    bg_subtracted = cv2.bitwise_and(orig_img, orig_img, mask=fg_mask_clean)

    print("[-] Background subtracted successfully based on HSV color levels.")

    # Step 7: Create comparison view
    comparison = create_comparison(orig_img, bg_subtracted, title1="Original Image", title2="Background Subtracted")

    # Step 8: Display result
    display_and_wait("Experiment 36 - Background Subtraction", comparison)
    return bg_subtracted

if __name__ == "__main__":
    subtract_background_by_color()
