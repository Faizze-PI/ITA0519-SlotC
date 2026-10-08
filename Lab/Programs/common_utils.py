"""
SIMATS Engineering - ITA0519 Computer Vision Lab
Common Utilities Module
-----------------------------------------------
Provides robust asset path resolution, standardized image loading,
side-by-side display formatting, and clean window management.
"""

import os
import sys
import cv2
import numpy as np

# Base paths
PROGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(PROGRAMS_DIR, ".."))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
VIDEOS_DIR = os.path.join(ASSETS_DIR, "videos")
CASCADES_DIR = os.path.join(ASSETS_DIR, "cascades")

def get_asset_path(subfolder, filename):
    """Resolve absolute path to an asset file safely."""
    path = os.path.join(ASSETS_DIR, subfolder, filename)
    if not os.path.exists(path):
        # Fallback to local assets if running elsewhere
        alt_path = os.path.join("assets", subfolder, filename)
        if os.path.exists(alt_path):
            return os.path.abspath(alt_path)
    return path

def load_image(filename="sample.jpg", flags=cv2.IMREAD_COLOR):
    """
    Load an image from assets/images with verification and fallback.
    """
    path = get_asset_path("images", filename)
    img = cv2.imread(path, flags)
    if img is None:
        print(f"[!] Warning: Could not load image from '{path}'. Creating synthetic test image.")
        # Create a 400x500 synthetic test image
        img = np.full((400, 500, 3), 240, dtype=np.uint8)
        cv2.putText(img, "SIMATS TEST IMAGE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
        if flags == cv2.IMREAD_GRAYSCALE:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def create_comparison(img1, img2, title1="Original", title2="Processed"):
    """
    Creates a standardized side-by-side comparison image with clear headers.
    Handles converting single-channel (grayscale) to 3-channel for concatenation.
    """
    # Normalize channels
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Normalize heights
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2)
    
    if h1 != target_h:
        img1 = cv2.resize(img1, (int(w1 * (target_h / h1)), target_h))
    if h2 != target_h:
        img2 = cv2.resize(img2, (int(w2 * (target_h / h2)), target_h))

    # Add header bars
    bar_h = 35
    header1 = np.full((bar_h, img1.shape[1], 3), (40, 40, 40), dtype=np.uint8)
    header2 = np.full((bar_h, img2.shape[1], 3), (40, 40, 40), dtype=np.uint8)
    
    cv2.putText(header1, title1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(header2, title2, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    panel1 = np.vstack([header1, img1])
    panel2 = np.vstack([header2, img2])
    
    # Concatenate with a vertical separator line
    separator = np.full((panel1.shape[0], 4, 3), (255, 255, 255), dtype=np.uint8)
    return np.hstack([panel1, separator, panel2])

def display_and_wait(window_name, image):
    """
    Standard window display with proper sizing and clean destruction.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, image)
    print(f"[*] Displaying '{window_name}'. Press any key (or 'q' / ESC) to close.")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return key
