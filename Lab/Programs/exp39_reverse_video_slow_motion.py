"""
========================================================================================
EXPERIMENT 39: PLAY VIDEO IN REVERSE MODE IN SLOW MOTION
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To write a Python function to read a captured video, reverse the chronological
    order of its frames, and play the video backwards in slow motion using OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    This experiment combines two fundamental video processing concepts:
    1. Temporal Inversion (Video Reversal):
       Reverses the frame sequence array:
           V_reversed = [ F_{N-1}, F_{N-2}, ..., F_1, F_0 ]
    2. Temporal Dilation (Slow Motion):
       Extends the presentation duration of each frame by scaling the inter-frame delay:
           Delay_slow = int((1000 / FPS) * slow_factor)
       For a 25 FPS video (nominal 40 ms) with a 3.0x slow-motion factor:
           Delay_slow = 40 ms * 3.0 = 120 ms per frame.

INPUT:
    assets/videos/sample_traffic.mp4

OUTPUT:
    Interactive OpenCV window rendering the video in reversed chronological sequence
    at 0.33x speed (slow motion).
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import get_asset_path

def play_reverse_slow_motion(video_name="sample_traffic.mp4", slow_factor=3.0):
    print(f"[*] Running Experiment 39: Reverse Video in Slow Motion (Factor: {slow_factor}x)")
    video_path = get_asset_path("videos", video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video at '{video_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    nominal_delay = int(1000 / fps)
    slow_delay = int(nominal_delay * slow_factor)

    # Step 1: Read all frames into buffer
    frames = []
    print("[-] Buffering video frames...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    print(f"[-] Buffered {total_frames} frames. Frame delay set to {slow_delay} ms (Slow Motion).")

    # Step 2: Reverse frames
    reversed_frames = frames[::-1]

    # Step 3: Play in reverse mode in slow motion
    window_name = "Experiment 39 - Reverse Slow-Motion Playback"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("[*] Starting slow-motion reverse playback. Press 'q' or ESC to stop.")
    for idx, frame in enumerate(reversed_frames):
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # HUD Banner
        cv2.rectangle(display_frame, (10, 10), (450, 70), (0, 0, 0), -1)
        cv2.putText(display_frame, f"REVERSE SLOW-MO ({1.0/slow_factor:.2f}x SPEED)", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        cv2.putText(display_frame, f"Frame: {idx + 1} of {total_frames} | Delay: {slow_delay}ms", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(slow_delay) & 0xFF
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()
    print("[-] Slow-motion reverse playback finished.")

if __name__ == "__main__":
    play_reverse_slow_motion()
