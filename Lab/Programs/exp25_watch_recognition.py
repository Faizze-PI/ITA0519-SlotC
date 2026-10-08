"""
========================================================================================
EXPERIMENT 25: OBJECT RECOGNITION (WATCH DETECTION)
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To recognize and localize a wristwatch in an input image using general object
    recognition techniques (Normalized Cross-Correlation Template Matching) in OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Object recognition seeks to identify and locate target objects within a scene.
    In classical computer vision, Normalized Correlation Coefficient (TM_CCOEFF_NORMED)
    slides a template T across an image I to compute similarity scores in [-1.0, 1.0]:
        R(x, y) = sum_{x', y'} (T'(x', y') * I'(x + x', y + y')) /
                  sqrt( sum_{x', y'} T'(x', y')^2 * sum_{x', y'} I'(x + x', y + y')^2 )
    where T' and I' represent mean-centered pixel values.

    The global maximum location (max_loc) corresponds to the best geometric match:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    Key Advantages:
        - Invariant to global brightness and contrast offsets.
        - Highly robust for rigid planar object localization.

INPUT:
    assets/images/watch.jpg (Scene image)
    assets/images/watch_template.jpg (Watch dial template)

OUTPUT:
    Side-by-side display showing the isolated template and the recognized watch
    highlighted with a green bounding box and match confidence score.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image, get_asset_path, create_comparison, display_and_wait

def run_experiment(image_name="watch.jpg", template_name="watch_template.jpg"):
    print("[*] Running Experiment 25: Watch Object Recognition")

    # Step 1: Read scene image and watch template
    scene_img = load_image(image_name)
    template_path = get_asset_path("images", template_name)
    template_img = cv2.imread(template_path)
    
    if template_img is None:
        print("[!] Template image not found. Creating default crop from scene.")
        h, w = scene_img.shape[:2]
        template_img = scene_img[int(h*0.3):int(h*0.7), int(w*0.35):int(w*0.65)].copy()

    # Step 2: Convert both to grayscale for correlation
    gray_scene = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    t_h, t_w = gray_template.shape[:2]

    # Step 3: Apply Template Matching
    res = cv2.matchTemplate(gray_scene, gray_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + t_w, top_left[1] + t_h)

    print(f"[-] Best match localized at: {top_left} with Confidence = {max_val * 100:.2f}%")

    # Step 4: Draw bounding box on recognized object
    recognized_img = scene_img.copy()
    cv2.rectangle(recognized_img, top_left, bottom_right, (0, 255, 0), 3)
    cv2.putText(recognized_img, f"WATCH DETECTED ({max_val*100:.1f}%)",
                (top_left[0], max(25, top_left[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Step 5: Format comparison view
    comparison = create_comparison(scene_img, recognized_img, title1="Original Scene", title2=f"Recognized Watch ({max_val*100:.1f}%)")

    # Display template standalone
    cv2.imshow("Experiment 25 - Watch Target Template", template_img)

    # Display recognition result
    display_and_wait("Experiment 25 - Watch Recognition", comparison)

if __name__ == "__main__":
    run_experiment()
