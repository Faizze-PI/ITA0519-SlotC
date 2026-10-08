"""
========================================================================================
EXPERIMENT 05: COLOR LEVEL HISTOGRAM ANALYSIS
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to analyze and plot the color histogram of a given
    input image across Blue, Green, and Red channels using OpenCV and Matplotlib.

THEORY & MATHEMATICAL FOUNDATION:
    A color histogram is a graphical representation of the tonal distribution in a
    digital color image. It plots the number of pixels for each tonal value across
    all color channels.
    
    In OpenCV, cv2.calcHist() computes the distribution:
        cv2.calcHist(images, channels, mask, histSize, ranges)
    where:
        - images: Source image in [img] format.
        - channels: Channel index ([0] for Blue, [1] for Green, [2] for Red).
        - mask: Mask image (None for full image analysis).
        - histSize: Number of bins ([256] for full 8-bit dynamic range).
        - ranges: Intensity range ([0, 256], exclusive upper bound).

INPUT:
    assets/images/sample.jpg

OUTPUT:
    Matplotlib window displaying the color image and its corresponding BGR histogram curves.
========================================================================================
"""

import os
import sys
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import load_image

def analyze_color_histogram(image_name="sample.jpg"):
    print("[*] Running Experiment 05: Color Level Histogram Analysis")

    # Step 1: Read the BGR image
    bgr_img = load_image(image_name)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Step 2: Compute histogram for each color channel
    colors = ('b', 'g', 'r')
    channel_names = ('Blue Channel', 'Green Channel', 'Red Channel')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 05: Color Level Histogram Analysis (SIMATS ITA0519)", fontsize=14, fontweight='bold')

    # Display image on left axis
    ax1.imshow(rgb_img)
    ax1.set_title("Input Image")
    ax1.axis('off')

    # Compute and plot histogram curves on right axis
    ax2.set_title("Color Level Distribution")
    ax2.set_xlabel("Pixel Intensity [0 - 255]")
    ax2.set_ylabel("Pixel Count")
    ax2.set_xlim([0, 256])
    ax2.grid(True, linestyle='--', alpha=0.6)

    for i, col in enumerate(colors):
        hist = cv2.calcHist([bgr_img], [i], None, [256], [0, 256])
        ax2.plot(hist, color=col, linewidth=2, label=channel_names[i])
        peak_intensity = hist.argmax()
        print(f"[-] {channel_names[i]}: Peak intensity at level {peak_intensity} (count: {int(hist[peak_intensity][0])})")

    ax2.legend(loc='upper right')
    plt.tight_layout()
    print("[*] Displaying Matplotlib Plot. Close the plot window to continue.")
    plt.show()

if __name__ == "__main__":
    analyze_color_histogram()
