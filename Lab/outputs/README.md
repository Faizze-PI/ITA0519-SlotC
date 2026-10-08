# Visual Output Gallery — ITA0519 Computer Vision Lab

**Saveetha Institute of Medical And Technical Sciences (SIMATS)**  
Department of Information Technology  
*Visual Execution Results & Program Output Gallery (Experiments 01 to 40)*

---

This gallery displays the direct graphical output captures and OpenCV visualization windows for all 40 laboratory experiments. Every program was executed and verified against actual input test media.

---

## 🖼️ Complete Visual Output Gallery (Exps 01 to 40)

| Exp # | Experiment Title & Concept | Visual Output Screenshot |
| :---: | :--- | :---: |
| **01** | **Convert Image to Grayscale**<br>Converts a 3-channel BGR digital image into an 8-bit single-channel grayscale representation using weighted luminance ($Y = 0.114B + 0.587G + 0.299R$). | ![Exp 01 Output](./exp01_output.png) |
| **02** | **Gaussian Blur Smoothing**<br>Performs 2D Gaussian low-pass spatial filtering ($11 \times 11$ kernel) to attenuate high-frequency noise while smoothing gradients. | ![Exp 02 Output](./exp02_output.png) |
| **03** | **Canny Outline Edge Detection**<br>Extracts clean, 1-pixel wide continuous object outlines using Canny multi-stage gradient analysis and hysteresis thresholding. | ![Exp 03 Output](./exp03_output.png) |
| **04** | **Histogram Equalization & Comparison**<br>Enhances image contrast by linearizing the Cumulative Distribution Function (CDF) across the full dynamic range [0 - 255]. | ![Exp 04 Output](./exp04_output.png) |
| **05** | **Color Level Histogram Analysis**<br>Analyzes and plots discrete tonal intensity distributions independently across Blue, Green, and Red channels. | ![Exp 05 Output](./exp05_output.png) |
| **06** | **Basic Image Erosion**<br>Applies morphological erosion with a $5 \times 5$ flat structuring element to peel away outer foreground boundaries and detach noise. | ![Exp 06 Output](./exp06_output.png) |
| **07** | **Video Playback Speed Control (Slow & Fast Motion)**<br>Manipulates frame presentation delay intervals to deliver smooth interactive Slow Motion (0.3x) and Fast Motion (3.0x) modes with HUD. | ![Exp 07 Output](./exp07_output.png) |
| **08** | **Basic Image Dilation**<br>Applies morphological dilation with a $5 \times 5$ structuring element to expand foreground regions and bridge subtle fractures. | ![Exp 08 Output](./exp08_output.png) |
| **09** | **Image Scaling (Bigger and Smaller)**<br>Demonstrates geometric coordinate resampling using `INTER_AREA` for downscaling (0.5x) and `INTER_CUBIC` for upscaling (1.3x). | ![Exp 09 Output](./exp09_output.png) |
| **10** | **90° Clockwise Rotation**<br>Performs an orthogonal 90-degree clockwise image plane transformation ($x' = H - 1 - y, y' = x$). | ![Exp 10 Output](./exp10_output.png) |
| **11** | **180° Clockwise Rotation**<br>Inverts both spatial axes simultaneously ($x' = W - 1 - x, y' = H - 1 - y$), generating an upside-down transformation. | ![Exp 11 Output](./exp11_output.png) |
| **12** | **270° Clockwise Rotation**<br>Performs a 270-degree clockwise rotation (equivalent to 90 degrees counter-clockwise: $x' = y, y' = W - 1 - x$). | ![Exp 12 Output](./exp12_output.png) |
| **13** | **Affine Transformation (3-Point Mapping)**<br>Computes a $2 \times 3$ affine mapping matrix from three anchor coordinate correspondences, preserving line parallelism. | ![Exp 13 Output](./exp13_output.png) |
| **14** | **Perspective Transformation (Homography)**<br>Computes a $3 \times 3$ perspective homography matrix from 4 quadrilateral points to produce a rectified bird's-eye view. | ![Exp 14 Output](./exp14_output.png) |
| **15** | **Harris Corner Detection**<br>Evaluates local structure tensors and eigenvalue response measures $R$ to detect and highlight corner feature junctions in red. | ![Exp 15 Output](./exp15_output.png) |
| **16** | **Sobel Gradient Filtering (X, Y, Combined)**<br>Calculates directional horizontal ($G_x$) and vertical ($G_y$) derivatives using $3 \times 3$ kernels to construct total gradient maps. | ![Exp 16 Output](./exp16_output.png) |
| **17** | **Digital Watermarking Technique**<br>Embeds a semi-transparent watermark emblem into a target host Region of Interest (ROI) using weighted linear alpha blending. | ![Exp 17 Output](./exp17_output.png) |
| **18** | **ROI Cropping, Copying & Pasting**<br>Extracts a rectangular sub-array Region of Interest (ROI) via NumPy slicing and clones it onto new destination coordinates. | ![Exp 18 Output](./exp18_output.png) |
| **19** | **Morphological Erosion with Structuring Elements**<br>Evaluates boundary erosion characteristics across Rectangular, Cross, and Elliptical structuring elements. | ![Exp 19 Output](./exp19_output.png) |
| **20** | **Morphological Dilation with Structuring Elements**<br>Evaluates foreground expansion and hole-filling across Rectangular, Cross, and Elliptical structuring elements. | ![Exp 20 Output](./exp20_output.png) |
| **21** | **Morphological Opening (Noise Elimination)**<br>Executes erosion followed by dilation ($A \circ B$) to eradicate isolated foreground noise dots without reducing object size. | ![Exp 21 Output](./exp21_output.png) |
| **22** | **Morphological Closing (Hole Filling)**<br>Executes dilation followed by erosion ($A \bullet B$) to bridge interior holes, cracks, and narrow gaps. | ![Exp 22 Output](./exp22_output.png) |
| **23** | **Morphological Top Hat Operation**<br>Subtracts the opened image from the original ($A - (A \circ B)$) to isolate high-frequency features brighter than their local background. | ![Exp 23 Output](./exp23_output.png) |
| **24** | **Morphological Black Hat Operation**<br>Subtracts the original image from the closed image ($(A \bullet B) - A$) to isolate dark features against bright backgrounds. | ![Exp 24 Output](./exp24_output.png) |
| **25** | **Object Recognition (Watch Detection)**<br>Applies Normalized Cross-Correlation Template Matching (`TM_CCOEFF_NORMED`) to detect and bound a target wristwatch with 99.9% confidence. | ![Exp 25 Output](./exp25_output.png) |
| **26** | **Video Frame Reversal**<br>Buffers video frames into memory, reverses sequence indexing ($F_{N-1} \to F_0$), and encodes a reverse-mode playback stream. | ![Exp 26 Output](./exp26_output.png) |
| **27** | **Human Face Detection (Haar Cascade)**<br>Applies Viola-Jones attentional cascade classifiers to detect and localize human faces with green bounding boxes. | ![Exp 27 Output](./exp27_output.png) |
| **28** | **Vehicle Detection in Video**<br>Detects and tracks moving vehicles across sequential video frames using Haar car cascades and adaptive MOG2 motion segmentation. | ![Exp 28 Output](./exp28_output.png) |
| **29** | **Human Eye Detection in Face ROI**<br>Implements a hierarchical cascade that isolates facial ROIs and bounds human eyes in blue rectangles. | ![Exp 29 Output](./exp29_output.png) |
| **30** | **Human Smile Detection in Face ROI**<br>Implements a hierarchical cascade that isolates lower-facial mouth regions to identify smiles. | ![Exp 30 Output](./exp30_output.png) |
| **31** | **Threshold-based Image Segmentation**<br>Partitions grayscale images into binary foreground and background masks using manual thresholds and Otsu's optimal variance method. | ![Exp 31 Output](./exp31_output.png) |
| **32** | **Synthetic Canvas with 4 Colored Corners**<br>Constructs a user-dimensioned white 3D NumPy array containing Black, Blue, Green, and Red boxes at each corner (1/10th size). | ![Exp 32 Output](./exp32_output.png) |
| **33** | **Synthetic Canvas with Rectangle Shape**<br>Creates a white digital canvas and renders a geometric centered rectangle with solid accents and borders. | ![Exp 33 Output](./exp33_output.png) |
| **34** | **Synthetic Canvas with Circle Shape**<br>Creates a white digital canvas and draws anti-aliased concentric circles using `cv2.circle()`. | ![Exp 34 Output](./exp34_output.png) |
| **35** | **Text String Overlay on Image**<br>Calculates font typography bounds using `cv2.getTextSize()` and renders a centered, high-contrast banner with user text. | ![Exp 35 Output](./exp35_output.png) |
| **36** | **Subtract Background by Color Levels**<br>Applies HSV color thresholding (`cv2.inRange`) to subtract background hues and retain foreground subjects. | ![Exp 36 Output](./exp36_output.png) |
| **37** | **Subtract Foreground by Color Levels**<br>Applies inverted HSV color masking to eliminate specific foreground objects while preserving the background. | ![Exp 37 Output](./exp37_output.png) |
| **38** | **Count Number of Faces in Image**<br>Applies multi-scale face detection, enumerates each detected bounding box (`#1`, `#2`, `#3`), and displays the total face count badge. | ![Exp 38 Output](./exp38_output.png) |
| **39** | **Play Video in Reverse Mode in Slow Motion**<br>Combines temporal frame array reversal with dilated delay intervals (120ms) for retrograde 0.33x slow-motion playback. | ![Exp 39 Output](./exp39_output.png) |
| **40** | **Text Extraction from Video (OCR)**<br>Samples video keyframes, preprocesses contrast, and applies Tesseract LSTM OCR to extract textual scene titles. | ![Exp 40 Output](./exp40_output.png) |
