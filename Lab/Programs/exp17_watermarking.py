"""
========================================================================================
EXPERIMENT 17: DIGITAL IMAGE WATERMARKING TECHNIQUE
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To design and implement a digital watermarking technique to effectively insert
    a watermark/logo into an original image using alpha blending in OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Digital image watermarking embeds copyright, verification, or branding data
    into a host image. Effective watermarking ensures the mark is clearly visible
    without obstructing the underlying visual content.
    
    This is achieved using linear alpha blending over a specific Region of Interest (ROI):
        Output(x, y) = alpha * Logo(x, y) + (1 - alpha) * Original_ROI(x, y) + gamma
    where:
        - alpha is the opacity weight of the watermark [0.0 - 1.0].
        - (1 - alpha) is the weight of the host image.
        - gamma is a scalar brightness offset (typically 0).

    In OpenCV:
        cv2.addWeighted(roi, 1 - alpha, watermark, alpha, gamma)

INPUT:
    assets/images/sample.jpg (Host image)
    assets/images/watermark_logo.png (Watermark emblem)

OUTPUT:
    Side-by-side display of the original image and the watermarked composite image.
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, get_asset_path, create_comparison, display_and_wait

def run_experiment(image_name="sample.jpg", logo_name="watermark_logo.png", alpha=0.35):
    print(f"[*] Running Experiment 17: Digital Watermarking (Alpha={alpha})")

    # Step 1: Read host image
    host_img = load_image(image_name)
    h_h, h_w = host_img.shape[:2]

    # Step 2: Read watermark logo
    logo_path = get_asset_path("images", logo_name)
    logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo_img is None:
        # Fallback: create a quick text badge
        logo_img = np.zeros((100, 150, 3), dtype=np.uint8)
        cv2.putText(logo_img, "SIMATS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Resize logo appropriately (e.g. 120x120)
    logo_size = 120
    logo_resized = cv2.resize(logo_img[:, :, :3], (logo_size, logo_size))

    # Step 3: Determine ROI position (Bottom-Right corner with 20px padding)
    y_offset = h_h - logo_size - 20
    x_offset = h_w - logo_size - 20

    watermarked_img = host_img.copy()
    roi = watermarked_img[y_offset:y_offset + logo_size, x_offset:x_offset + logo_size]

    # Step 4: Blend watermark into ROI
    blended_roi = cv2.addWeighted(roi, 1.0 - alpha, logo_resized, alpha, 0)
    watermarked_img[y_offset:y_offset + logo_size, x_offset:x_offset + logo_size] = blended_roi

    # Also add a semi-transparent text copyright banner across the bottom
    banner = watermarked_img.copy()
    cv2.rectangle(banner, (0, h_h - 25), (h_w, h_h), (0, 0, 0), -1)
    cv2.addWeighted(banner, 0.5, watermarked_img, 0.5, 0, watermarked_img)
    cv2.putText(watermarked_img, "Protected by SIMATS Engineering - ITA0519", (15, h_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    print("[-] Watermark successfully embedded into host image.")

    # Step 5: Side-by-side comparison
    comparison = create_comparison(host_img, watermarked_img, title1="Original Host Image", title2="Watermarked Image")

    # Step 6: Display result
    display_and_wait("Experiment 17 - Digital Watermarking", comparison)

if __name__ == "__main__":
    run_experiment()
