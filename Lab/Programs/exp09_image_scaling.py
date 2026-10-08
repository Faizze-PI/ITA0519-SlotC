"""
========================================================================================
EXPERIMENT 09: IMAGE SCALING (RESIZING TO BIGGER AND SMALLER SIZES)
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement image scaling techniques to resize an image to both smaller and
    larger dimensions using various interpolation methods in OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Image scaling (resampling) involves mapping discrete pixel coordinates from a source
    grid to a new destination grid:
        x' = s_x * x,   y' = s_y * y

    Because new coordinates rarely align with integer pixel centers, interpolation is required:
    1. cv2.INTER_AREA: Pixel area relation resampling. Highly recommended for downscaling
       (shrinking) to prevent moire patterns and preserve high-contrast edges.
    2. cv2.INTER_LINEAR: Bilinear interpolation using a 2x2 neighborhood. Standard, fast default.
    3. cv2.INTER_CUBIC: Bicubic interpolation using a 4x4 neighborhood. Recommended for upscaling
       (enlarging) to generate smooth, crisp continuous gradients.

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Separate windows showing the Original, Downscaled (0.5x), and Upscaled (1.5x) images,
    along with terminal printouts of the respective resolutions.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, display_and_wait

def run_experiment(image_name="sample.jpg", down_factor=0.5, up_factor=1.5):
    print(f"[*] Running Experiment 09: Image Scaling (Down={down_factor}x, Up={up_factor}x)")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    h, w = orig_img.shape[:2]
    print(f"[-] Original Resolution: {w} x {h}")

    # Step 2: Downscale image (Smaller size) using INTER_AREA
    down_w = int(w * down_factor)
    down_h = int(h * down_factor)
    small_img = cv2.resize(orig_img, (down_w, down_h), interpolation=cv2.INTER_AREA)
    print(f"[-] Downscaled Resolution: {down_w} x {down_h} (Method: INTER_AREA)")

    # Step 3: Upscale image (Bigger size) using INTER_CUBIC
    up_w = int(w * up_factor)
    up_h = int(h * up_factor)
    big_img = cv2.resize(orig_img, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
    print(f"[-] Upscaled Resolution: {up_w} x {up_h} (Method: INTER_CUBIC)")

    # Display images
    cv2.imshow("Experiment 09 - 1. Original", orig_img)
    cv2.imshow("Experiment 09 - 2. Downscaled (Small)", small_img)
    cv2.imshow("Experiment 09 - 3. Upscaled (Big)", big_img)

    print("[*] Displaying all 3 scaled images. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_experiment()
