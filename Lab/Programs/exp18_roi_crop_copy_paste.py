"""
========================================================================================
EXPERIMENT 18: REGION OF INTEREST (ROI) CROPPING, COPYING AND PASTING
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To select a Region of Interest (ROI) from an input image, crop it, copy its data,
    and paste it onto a new location in the image using OpenCV and NumPy array slicing.

THEORY & MATHEMATICAL FOUNDATION:
    In digital image processing, an ROI is a sub-region of an image grid chosen for
    focused analysis. Because OpenCV represents images as contiguous NumPy n-dimensional
    arrays, sub-matrices can be extracted with zero overhead via multidimensional slicing:
        ROI = image[y_start : y_end, x_start : x_end]

    Key Operations:
        1. Cropping : Extracting the specified bounding sub-array.
        2. Copying  : Calling .copy() to decouple memory buffers.
        3. Pasting  : Assigning the extracted array to an identical dimensional destination slice:
           image[y_dest : y_dest + h, x_dest : x_dest + w] = ROI

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Multi-panel display showing:
    1. Original image annotated with ROI selection box.
    2. Cropped standalone ROI.
    3. Final image with pasted duplicate ROI.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg"):
    print("[*] Running Experiment 18: ROI Cropping, Copying, and Pasting")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    h, w = orig_img.shape[:2]

    # Step 2: Define ROI bounding coordinates (e.g. house/sun in sample image)
    # y: 150 to 320, x: 180 to 360
    y1, y2 = int(h * 0.35), int(h * 0.80)
    x1, x2 = int(w * 0.30), int(w * 0.60)
    roi_h = y2 - y1
    roi_w = x2 - x1

    print(f"[-] Selected ROI coordinates: X=[{x1}, {x2}], Y=[{y1}, {y2}] (Dimensions: {roi_w}x{roi_h})")

    # Step 3: Crop and Copy ROI
    cropped_roi = orig_img[y1:y2, x1:x2].copy()

    # Step 4: Paste ROI into a new target position (Top-Left corner with border)
    pasted_img = orig_img.copy()
    dst_y, dst_x = 20, 20
    # Ensure paste fits inside canvas
    dst_y2 = min(dst_y + roi_h, h)
    dst_x2 = min(dst_x + roi_w, w)
    paste_h = dst_y2 - dst_y
    paste_w = dst_x2 - dst_x

    pasted_img[dst_y:dst_y2, dst_x:dst_x2] = cropped_roi[:paste_h, :paste_w]
    # Draw border around pasted clone
    cv2.rectangle(pasted_img, (dst_x, dst_y), (dst_x2, dst_y2), (0, 255, 255), 2)
    cv2.putText(pasted_img, "Pasted ROI", (dst_x + 5, dst_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Step 5: Annotate original image showing source ROI bounding box
    annotated_orig = orig_img.copy()
    cv2.rectangle(annotated_orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(annotated_orig, "Selected ROI", (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Step 6: Create comparison
    comparison = create_comparison(annotated_orig, pasted_img, title1="Source ROI Selected", title2="Cloned ROI Pasted")

    # Display standalone cropped ROI
    cv2.imshow("Experiment 18 - Standalone Cropped ROI", cropped_roi)

    # Display before-and-after comparison
    display_and_wait("Experiment 18 - ROI Copy and Paste", comparison)

if __name__ == "__main__":
    run_experiment()
