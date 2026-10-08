"""
========================================================================================
EXPERIMENT 37: FOREGROUND SUBTRACTION BASED ON COLOR LEVELS
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to subtract/remove the foreground objects of an input
    image based on specific color levels in the HSV color space using OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Foreground subtraction isolates and eliminates specific target colored objects
    from an image while preserving the background environment.
    
    Algorithmic Steps:
    1. HSV Conversion: Converts BGR image into HSV color space.
    2. Foreground Color Masking:
       Detect the specific foreground color signature (e.g. green circle and red rectangle):
           fg_mask_green = cv2.inRange(hsv, lower_green, upper_green)
           fg_mask_red   = cv2.inRange(hsv, lower_red, upper_red)
           fg_mask_total = cv2.bitwise_or(fg_mask_green, fg_mask_red)
    3. Background Retention Mask:
       Invert the foreground mask to obtain the background mask:
           bg_mask = cv2.bitwise_not(fg_mask_total)
    4. Bitwise Subtraction:
       Extract background-only regions by masking out the foreground:
           result = cv2.bitwise_and(img, img, mask=bg_mask)

INPUT:
    assets/images/color_object.jpg

OUTPUT:
    Side-by-side comparison of the original image and the foreground-subtracted image.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def subtract_foreground_by_color(image_name="color_object.jpg"):
    print("[*] Running Experiment 37: Subtract Foreground Based on Color Levels")

    # Step 1: Read input image
    orig_img = load_image(image_name)

    # Step 2: Convert to HSV color space
    hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    # Step 3: Define HSV ranges for foreground objects (Green object + Red object)
    # Green object: Hue ~ 35 - 85
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Red object: Hue wraps around 0-10 and 170-180
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    # Total foreground mask
    fg_mask = cv2.bitwise_or(mask_green, mask_red)

    # Step 4: Invert foreground mask to preserve only background
    bg_mask = cv2.bitwise_not(fg_mask)

    # Clean mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bg_mask_clean = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)

    # Step 5: Extract image with foreground removed (subtracted)
    fg_subtracted = cv2.bitwise_and(orig_img, orig_img, mask=bg_mask_clean)

    print("[-] Foreground subtracted successfully based on HSV color levels.")

    # Step 6: Create comparison view
    comparison = create_comparison(orig_img, fg_subtracted, title1="Original Image", title2="Foreground Subtracted")

    # Step 7: Display result
    display_and_wait("Experiment 37 - Foreground Subtraction", comparison)
    return fg_subtracted

if __name__ == "__main__":
    subtract_foreground_by_color()
