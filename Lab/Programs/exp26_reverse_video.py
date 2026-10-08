"""
========================================================================================
EXPERIMENT 26: VIDEO FRAME REVERSAL
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To implement a Python function to extract, reverse, and save/display the frames
    of a video file to produce a reverse-mode video playback using OpenCV.

THEORY & MATHEMATICAL FOUNDATION:
    Digital video playback is fundamentally a time-indexed series of 2D images:
        V = [ F_0, F_1, F_2, ..., F_{N-1} ]
    Reversing the temporal progression of frames creates a retrograde playback:
        V_rev = [ F_{N-1}, F_{N-2}, ..., F_1, F_0 ]

    Process Flow:
        1. Sequential Capture: Read all frames from cv2.VideoCapture into an in-memory buffer.
        2. Sequence Reversal : Apply Python slice reversal: frames[::-1].
        3. Video Serialization: Initialize cv2.VideoWriter with identical codec, dimensions,
           and frame rate, then sequentially encode reversed frames to disk.
        4. Interactive Playback: Render the reversed frames in an active display window.

INPUT:
    assets/videos/sample_traffic.mp4

OUTPUT:
    Saved reversed video file (assets/videos/reversed_output.mp4) and real-time playback.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import get_asset_path, VIDEOS_DIR

def reverse_video(video_name="sample_traffic.mp4", save_output=True):
    print(f"[*] Running Experiment 26: Reverse Video Frame Processing ({video_name})")
    video_path = get_asset_path("videos", video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Unable to open video at '{video_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Step 1: Read all frames into memory
    frames = []
    print("[-] Reading video frames into buffer...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    print(f"[-] Successfully buffered {total_frames} frames ({w}x{h} at {fps:.1f} FPS).")

    # Step 2: Reverse frame sequence
    reversed_frames = frames[::-1]
    print("[-] Frames reversed in temporal order.")

    # Step 3: Optionally save reversed video to disk
    if save_output:
        out_path = os.path.join(VIDEOS_DIR, "reversed_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in reversed_frames:
            out.write(f)
        out.release()
        print(f"[+] Saved reversed video to: {out_path}")

    # Step 4: Display reversed playback
    window_name = "Experiment 26 - Reverse Video Playback"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    delay = int(1000 / fps)

    print("[*] Playing reversed video. Press 'q' or ESC to stop.")
    for idx, frame in enumerate(reversed_frames):
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, f"REVERSE FRAME: {idx + 1}/{total_frames}", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(delay) & 0xFF
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()
    print("[-] Playback completed.")

if __name__ == "__main__":
    reverse_video()
