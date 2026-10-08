"""
========================================================================================
EXPERIMENT 35: TEXT STRING OVERLAY ON IMAGE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to overlay a custom text string entered by the user
    onto a given image using OpenCV's putText function.

THEORY & MATHEMATICAL FOUNDATION:
    Text overlay is critical for HUD displays, watermarking, timestamps, and bounding
    box classification labels in computer vision pipelines.
    
    In OpenCV, cv2.putText() renders vector Hershey fonts:
        cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)
    where:
        - text: The string sequence to render.
        - org: Bottom-left coordinate (x, y) where the text baseline starts.
        - fontFace: Font typography (e.g. cv2.FONT_HERSHEY_SIMPLEX).
        - fontScale: Multiplier scaling font size.
        - color: BGR tuple for text font color.
        - thickness: Stroke weight in pixels.
        - lineType: Set to cv2.LINE_AA for anti-aliased, smooth glyph rendering.

    Text Centering via cv2.getTextSize():
        Calculates exact pixel width and height of the rendered string:
            (w_text, h_text), baseline = cv2.getTextSize(text, font, scale, thickness)
            org_x = (img_width - w_text) // 2
            org_y = (img_height + h_text) // 2

INPUT:
    assets/images/sample.jpg
    User-specified text string (Default: "SIMATS COMPUTER VISION LAB - ITA0519")

OUTPUT:
    Side-by-side comparison of the original image and the text-annotated image.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, create_comparison, display_and_wait

def overlay_text_on_image(image_name="sample.jpg", user_text="SIMATS COMPUTER VISION LAB - ITA0519"):
    print(f"[*] Running Experiment 35: Text Overlay: \"{user_text}\"")

    # Step 1: Read input image
    orig_img = load_image(image_name)
    h, w = orig_img.shape[:2]

    annotated_img = orig_img.copy()

    # Step 2: Calculate text dimensions for centering
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(user_text, font, font_scale, thickness)

    # Calculate centered baseline coordinates
    org_x = max(10, (w - text_w) // 2)
    org_y = h // 2

    # Step 3: Draw a semi-transparent dark banner behind text for high legibility
    pad = 12
    banner = annotated_img.copy()
    cv2.rectangle(banner, (org_x - pad, org_y - text_h - pad),
                  (min(w, org_x + text_w + pad), org_y + baseline + pad), (0, 0, 0), -1)
    cv2.addWeighted(banner, 0.6, annotated_img, 0.4, 0, annotated_img)

    # Step 4: Render text string with anti-aliasing
    cv2.putText(annotated_img, user_text, (org_x, org_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    # Step 5: Format comparison view
    comparison = create_comparison(orig_img, annotated_img, title1="Original Image", title2="Image with User Text Overlay")

    # Step 6: Display result
    display_and_wait("Experiment 35 - Text String Overlay", comparison)
    return annotated_img

def run_experiment():
    default_text = "SIMATS COMPUTER VISION LAB - ITA0519"
    try:
        if sys.stdin.isatty():
            t_input = input(f"Enter text string to overlay [Default '{default_text}']: ").strip()
            user_text = t_input if t_input else default_text
        else:
            user_text = default_text
    except Exception:
        user_text = default_text

    overlay_text_on_image("sample.jpg", user_text)

if __name__ == "__main__":
    run_experiment()
