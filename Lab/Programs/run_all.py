"""
========================================================================================
SIMATS ENGINEERING - SAVEETHA INSTITUTE OF MEDICAL AND TECHNICAL SCIENCES
DEPARTMENT OF INFORMATION TECHNOLOGY
ITA0519: COMPUTER VISION & DIGITAL IMAGE PROCESSING LAB
========================================================================================
Interactive Experiment Runner (Experiments 01 to 40)
----------------------------------------------------------------------------------------
Allows students and faculty evaluators to execute any lab experiment by entering its
number (1 - 40), or run an automated test sweep across all experiments.
========================================================================================
"""

import os
import sys
import subprocess

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROGRAMS_DIR = os.path.join(BASE_DIR, "programs")

EXPERIMENTS = {
    1:  ("exp01_grayscale.py", "Convert Image to Grayscale"),
    2:  ("exp02_gaussian_blur.py", "Gaussian Blur Smoothing"),
    3:  ("exp03_canny_outline.py", "Canny Edge Outline Detection"),
    4:  ("exp04_histogram_equalization.py", "Histogram Equalization & Comparison"),
    5:  ("exp05_color_histogram.py", "Color Level Histogram Analysis"),
    6:  ("exp06_image_erosion.py", "Basic Image Erosion"),
    7:  ("exp07_video_speed_control.py", "Video Playback Speed Control (Slow/Fast)"),
    8:  ("exp08_image_dilation.py", "Basic Image Dilation"),
    9:  ("exp09_image_scaling.py", "Image Scaling (Bigger and Smaller)"),
    10: ("exp10_rotate_90_clockwise.py", "90-Degree Clockwise Rotation"),
    11: ("exp11_rotate_180_clockwise.py", "180-Degree Clockwise Rotation"),
    12: ("exp12_rotate_270_clockwise.py", "270-Degree Clockwise Rotation"),
    13: ("exp13_affine_transformation.py", "Affine Transformation"),
    14: ("exp14_perspective_transformation.py", "Perspective Transformation"),
    15: ("exp15_harris_corner_detection.py", "Harris Corner Detection"),
    16: ("exp16_sobel_filter.py", "Sobel Algorithm Image Filtering"),
    17: ("exp17_watermarking.py", "Digital Watermarking Technique"),
    18: ("exp18_roi_crop_copy_paste.py", "ROI Cropping, Copying & Pasting"),
    19: ("exp19_morphological_erosion.py", "Morphological Erosion with SEs"),
    20: ("exp20_morphological_dilation.py", "Morphological Dilation with SEs"),
    21: ("exp21_morphological_opening.py", "Morphological Opening Technique"),
    22: ("exp22_morphological_closing.py", "Morphological Closing Technique"),
    23: ("exp23_morphological_tophat.py", "Morphological Top Hat Operation"),
    24: ("exp24_morphological_blackhat.py", "Morphological Black Hat Operation"),
    25: ("exp25_watch_recognition.py", "Object Recognition (Watch Detection)"),
    26: ("exp26_reverse_video.py", "Video Frame Reversal"),
    27: ("exp27_face_detection.py", "Human Face Detection (Haar Cascade)"),
    28: ("exp28_vehicle_detection_video.py", "Vehicle Detection in Video"),
    29: ("exp29_eye_detection.py", "Human Eye Detection in Face ROI"),
    30: ("exp30_smile_detection.py", "Human Smile Detection in Face ROI"),
    31: ("exp31_threshold_segmentation.py", "Threshold-based Image Segmentation"),
    32: ("exp32_canvas_four_corners.py", "White Canvas with 4 Colored Corners"),
    33: ("exp33_canvas_rectangle.py", "White Canvas with Rectangle Shape"),
    34: ("exp34_canvas_circle.py", "White Canvas with Circle Shape"),
    35: ("exp35_canvas_text.py", "Text String Overlay on Image"),
    36: ("exp36_subtract_background_color.py", "Subtract Background by Color Levels"),
    37: ("exp37_subtract_foreground_color.py", "Subtract Foreground by Color Levels"),
    38: ("exp38_count_faces.py", "Count Number of Faces in Image"),
    39: ("exp39_reverse_video_slow_motion.py", "Reverse Video in Slow Motion"),
    40: ("exp40_video_text_extraction.py", "Text Extraction from Video (OCR)")
}

def print_menu():
    print("\n" + "=" * 80)
    print("      SIMATS ENGINEERING - ITA0519 COMPUTER VISION LAB EXPERIMENTS")
    print("=" * 80)
    
    col1 = [i for i in range(1, 21)]
    col2 = [i for i in range(21, 41)]
    
    for i, j in zip(col1, col2):
        file1, title1 = EXPERIMENTS[i]
        file2, title2 = EXPERIMENTS[j]
        left = f"[{i:02d}] {title1[:32]:<32}"
        right = f"[{j:02d}] {title2[:32]:<32}"
        print(f"  {left}  |  {right}")

    print("=" * 80)
    print("  Special Commands: [V]erify All  |  [Q]uit")
    print("=" * 80)

def run_experiment(exp_num):
    if exp_num not in EXPERIMENTS:
        print(f"[!] Invalid experiment number: {exp_num}. Choose 1 to 40.")
        return

    script_name, title = EXPERIMENTS[exp_num]
    script_path = os.path.join(PROGRAMS_DIR, script_name)

    print("\n" + "-" * 80)
    print(f"[*] LAUNCHING EXPERIMENT {exp_num:02d}: {title}")
    print(f"[*] SCRIPT: {script_path}")
    print("-" * 80)

    try:
        subprocess.run([sys.executable, script_path], cwd=PROGRAMS_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Experiment exited with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
    print("-" * 80)

def verify_all():
    print("\n[*] Starting automated verification across all 40 experiment modules...")
    success_count = 0
    failed = []

    for num in range(1, 41):
        script_name, title = EXPERIMENTS[num]
        script_path = os.path.join(PROGRAMS_DIR, script_name)
        # Test compile and import syntax
        res = subprocess.run([sys.executable, "-m", "py_compile", script_path], capture_output=True, text=True)
        if res.returncode == 0:
            success_count += 1
            print(f"  [OK] Exp {num:02d}: {script_name}")
        else:
            failed.append((num, script_name, res.stderr))
            print(f"  [FAIL] Exp {num:02d}: {script_name} - {res.stderr.strip()}")

    print("\n" + "=" * 50)
    print(f"Verification Results: {success_count}/40 Passed ({success_count*100//40}%)")
    print("=" * 50)

def main():
    while True:
        print_menu()
        choice = input("\nEnter experiment number to run (1-40), 'V' to verify, or 'Q' to quit: ").strip().lower()
        if choice in ('q', 'quit', 'exit'):
            print("Exiting runner. Goodbye!")
            break
        elif choice in ('v', 'verify'):
            verify_all()
        else:
            try:
                num = int(choice)
                run_experiment(num)
            except ValueError:
                print("[!] Please enter a valid number between 1 and 40.")

if __name__ == "__main__":
    main()
