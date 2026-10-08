"""
========================================================================================
EXPERIMENT 07: VIDEO SPEED CONTROL (SLOW MOTION AND FAST MOTION)
========================================================================================
Course   : ITA0519 - Computer Vision Lab
College  : SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences

AIM:
    To perform basic video processing operations: read a captured/stored video file
    and display its playback in normal, slow-motion, and fast-motion speeds.

THEORY & MATHEMATICAL FOUNDATION:
    A video is a sequential collection of discrete photographic frames displayed at a
    rate known as Frames Per Second (FPS).
    For a video recorded at nominal frame rate F (e.g., 25 FPS), the inter-frame delay
    in milliseconds is:
        T_normal = 1000 / F (e.g., 1000 / 25 = 40 ms)

    Playback rate is regulated by the millisecond argument passed to cv2.waitKey(delay):
        - Slow Motion (e.g., 0.3x speed): delay_slow = int(T_normal * 3.0) -> 120 ms
        - Fast Motion (e.g., 3.0x speed): delay_fast = max(1, int(T_normal / 3.0)) -> 13 ms

    Interactive Keyboard Controls in this program:
        [S] - Switch to Slow Motion (0.3x speed)
        [F] - Switch to Fast Motion (3.0x speed)
        [N] - Switch to Normal Motion (1.0x speed)
        [Q] / [ESC] - Quit Playback

INPUT:
    assets/videos/sample_traffic.mp4

OUTPUT:
    Interactive OpenCV window showing video frames with real-time speed overlay HUD.
========================================================================================
"""

import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import get_asset_path

def run_experiment(video_name="sample_traffic.mp4"):
    print("[*] Running Experiment 07: Video Processing (Slow Motion and Fast Motion)")
    video_path = get_asset_path("videos", video_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video file at '{video_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0
    normal_delay = int(1000 / fps)

    modes = {
        'NORMAL': (normal_delay, "Normal (1.0x)"),
        'SLOW': (int(normal_delay * 3.0), "Slow Motion (0.3x)"),
        'FAST': (max(1, int(normal_delay / 3.0)), "Fast Motion (3.0x)")
    }
    current_mode = 'NORMAL'

    print("[-] Video loaded successfully.")
    print("    Controls: Press [S] for Slow Motion, [F] for Fast Motion, [N] for Normal, [Q] to Quit.")

    window_name = "Experiment 07 - Video Playback Speed Control"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video continuously for uninterrupted lab demonstration
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret: break

        delay, label = modes[current_mode]

        # Draw HUD banner
        hud = frame.copy()
        cv2.rectangle(hud, (10, 10), (360, 85), (0, 0, 0), -1)
        cv2.addWeighted(hud, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"MODE: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Keys: [S]low  [F]ast  [N]ormal  [Q]uit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF

        if key in (ord('q'), 27): # 'q' or ESC
            break
        elif key == ord('s'):
            current_mode = 'SLOW'
            print("[*] Switched to SLOW MOTION")
        elif key == ord('f'):
            current_mode = 'FAST'
            print("[*] Switched to FAST MOTION")
        elif key == ord('n'):
            current_mode = 'NORMAL'
            print("[*] Switched to NORMAL MOTION")

    cap.release()
    cv2.destroyAllWindows()
    print("[-] Video playback completed.")

if __name__ == "__main__":
    run_experiment()
