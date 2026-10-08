"""
========================================================================================
EXPERIMENT 40: TEXT EXTRACTION FROM VIDEO USING OCR
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to extract and recognize textual strings embedded within
    a video stream using OpenCV keyframe processing and Optical Character Recognition (OCR).

THEORY & MATHEMATICAL FOUNDATION:
    Extracting text from dynamic video streams involves a multi-stage vision pipeline:
    1. Temporal Keyframe Sampling:
       Processing every single video frame is computationally wasteful and redundant.
       Sampling every k-th frame (e.g. every 15-20 frames) captures scene text transitions
       while drastically reducing latency.
    2. Image Pre-processing for OCR:
       - Grayscale conversion: Decouples text luminance from background color.
       - Thresholding / Binarization: Enhances contrast between character glyphs and background.
       - Morphological Denoising: Cleans character edges.
    3. Optical Character Recognition (Tesseract Engine):
       Converts character pixel bitmaps into unicode strings using LSTM neural networks
       and language modeling.
    4. Text De-duplication:
       Filters adjacent near-identical strings to produce a clean transcript log.

INPUT:
    assets/videos/text_video.mp4

OUTPUT:
    Console display of all unique extracted textual lines and an active video window
    showing the real-time recognized text overlay.
========================================================================================
"""

import os
import sys
import shutil
import cv2
import pytesseract

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import get_asset_path

# Configure Tesseract binary path
tesseract_candidate = shutil.which("tesseract")
if not tesseract_candidate:
    fallback_paths = [
        r"C:\Users\Faizze-PI\scoop\shims\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]
    for p in fallback_paths:
        if os.path.exists(p):
            tesseract_candidate = p
            break

if tesseract_candidate:
    pytesseract.pytesseract.tesseract_cmd = tesseract_candidate

def extract_text_from_video(video_name="text_video.mp4", sample_interval=20):
    print(f"[*] Running Experiment 40: Text Extraction from Video ({video_name})")
    print(f"[-] Configured Tesseract Engine: {pytesseract.pytesseract.tesseract_cmd}")
    video_path = get_asset_path("videos", video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video file at '{video_path}'.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[-] Video Loaded: {total_frames} total frames at {fps:.1f} FPS. Sampling every {sample_interval} frames.")

    extracted_lines = []
    seen_texts = set()

    frame_idx = 0
    window_name = "Experiment 40 - Video Text Extraction"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    current_detected_text = "Scanning..."

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_idx += 1

        # Perform OCR at specified sample intervals
        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # High-contrast thresholding for clean OCR
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

            # Run Tesseract OCR with page segmentation mode 6 (Assume a single uniform block of text)
            custom_config = r'--oem 3 --psm 6'
            try:
                raw_text = pytesseract.image_to_string(thresh, config=custom_config)
                clean_lines = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 3]

                for line in clean_lines:
                    if line not in seen_texts:
                        seen_texts.add(line)
                        extracted_lines.append((frame_idx, line))
                        print(f"    [Frame {frame_idx:03d} OCR] -> \"{line}\"")
                
                if clean_lines:
                    current_detected_text = clean_lines[-1]
            except Exception as e:
                print(f"[!] OCR Processing Note: {e}")

        # Render HUD on video playback
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (10, display_frame.shape[0] - 65), (display_frame.shape[1] - 10, display_frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(display_frame, f"LIVE OCR: {current_detected_text[:50]}",
                    (20, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(25) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n=======================================================")
    print("SUMMARY OF ALL TEXT EXTRACTED FROM VIDEO:")
    print("=======================================================")
    for f_idx, line in extracted_lines:
        print(f" - [Frame {f_idx:03d}]: {line}")
    print("=======================================================\n")

    return extracted_lines

if __name__ == "__main__":
    extract_text_from_video()
