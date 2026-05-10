
<div align="center">

# 🎯 Real-Time Object Detection & Tracking

### YOLOv8 + SORT Tracker | CPU-Optimized | Python

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Mac%20%7C%20Linux-0EA5E9?style=for-the-badge)]()
[![CPU](https://img.shields.io/badge/Hardware-CPU%20Only-F59E0B?style=for-the-badge&logo=intel&logoColor=white)]()

<br/>

> **Detect and track multiple objects in real-time using YOLOv8 and the SORT algorithm —
> no GPU required. Works with webcam or any video file.**

<br/>

![Object Detection Demo](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🧠 How It Works](#-how-it-works)
- [🗂️ Project Structure](#️-project-structure)
- [⚙️ Requirements](#️-requirements)
- [🚀 Quick Start](#-quick-start)
- [📓 Notebook Guide](#-notebook-guide)
- [🔧 Configuration](#-configuration)
- [📊 Output](#-output)
- [🐛 Common Errors & Fixes](#-common-errors--fixes)
- [💡 Customization](#-customization)
- [🙌 Acknowledgements](#-acknowledgements)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Detection Model** | YOLOv8 Nano — fastest YOLO model, optimized for CPU |
| 🔁 **Tracking Algorithm** | SORT (Simple Online and Realtime Tracking) with Kalman Filter |
| 📷 **Input Sources** | Webcam (live) or any `.mp4` / `.avi` video file |
| 🎨 **Visualizations** | Stylish bounding boxes, unique color per ID, HUD overlay |
| 💾 **Output** | Saves processed video + sample frame preview + class stats chart |
| ⚡ **CPU Friendly** | No GPU needed — runs on any modern laptop |
| 🏷️ **80 Object Classes** | People, cars, bikes, animals, furniture, and more |
| 📊 **Analytics** | Per-class detection count chart across all frames |

---

## 🧠 How It Works

```
Video Frame
    │
    ▼
┌─────────────────────┐
│   YOLOv8n Model     │  ← Detects objects + confidence scores
│   (Object Detector) │
└─────────┬───────────┘
          │  [x1,y1,x2,y2, conf, class]
          ▼
┌─────────────────────┐
│   SORT Tracker      │  ← Assigns unique IDs, tracks across frames
│   (Kalman Filter +  │     using Hungarian algorithm for matching
│   Hungarian Match)  │
└─────────┬───────────┘
          │  [x1,y1,x2,y2, track_id]
          ▼
┌─────────────────────┐
│   Draw & Display    │  ← Colored boxes, labels, IDs, FPS HUD
└─────────────────────┘
          │
          ▼
   output_tracked.mp4
```

**SORT Algorithm Steps:**
1. **Predict** — Each tracked object's next position is predicted using a Kalman Filter
2. **Match** — New detections are matched to existing tracks using IoU + Hungarian algorithm
3. **Update** — Matched tracks are updated; unmatched detections become new tracks
4. **Delete** — Tracks not seen for `max_age` frames are removed

---

## 🗂️ Project Structure

```
📁 object-detection-tracking/
│
├── 📓 Object_Detection_Tracking.ipynb   ← Main Jupyter notebook (run this!)
├── 🎬 sample_video.mp4                  ← Auto-downloaded on first run
├── 🎥 output_tracked.mp4               ← Generated output video
├── 🖼️  preview.png                      ← Sample frames preview image
├── 📊 detection_stats.png              ← Per-class detection bar chart
├── 📄 README.md                         ← You are here
│
└── 📁 venv/                             ← Virtual environment (not committed)
```

---

## ⚙️ Requirements

### System
- Python **3.10 or higher**
- Windows / macOS / Linux
- **No GPU required** — runs entirely on CPU

### Python Libraries

| Library | Version | Purpose |
|---|---|---|
| `ultralytics` | latest | YOLOv8 model |
| `opencv-python` | latest | Video reading & frame processing |
| `numpy` | latest | Numerical operations |
| `filterpy` | latest | Kalman Filter for SORT tracker |
| `scipy` | latest | Hungarian algorithm (linear assignment) |
| `lapx` | latest | Linear assignment problem solver |
| `matplotlib` | latest | Detection statistics chart |
| `Pillow` | latest | Image utilities |
| `jupyter` | latest | Notebook environment |

---

## 🚀 Quick Start

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/object-detection-tracking.git
cd object-detection-tracking
```

### Step 2 — Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate — Windows (PowerShell)
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

> ✅ You'll see `(venv)` at the start of your terminal line when it's active.

### Step 3 — Install Dependencies

```bash
pip install ultralytics opencv-python numpy filterpy scipy Pillow matplotlib lapx ipykernel jupyter
```

### Step 4 — Register Jupyter Kernel

```bash
python -m ipykernel install --user --name=venv --display-name "Python (object-detection)"
```

### Step 5 — Open the Notebook

```bash
code Object_Detection_Tracking.ipynb
```

> In VS Code: click the kernel selector (top right) → choose **"Python (object-detection)"**

### Step 6 — Run All Cells

Press `Ctrl + Alt + R` or click **⏩ Run All** at the top of the notebook.

---

## 📓 Notebook Guide

The notebook is organized into **10 steps**. Run them **top to bottom** in order:

| Step | Cell | What it does |
|---|---|---|
| 1 | Install | Installs all required packages (run once) |
| 2 | Imports | Loads all libraries into memory |
| 3 | SORT Tracker | Builds the Kalman Filter tracker from scratch |
| 4 | Load YOLO | Downloads & loads YOLOv8n model (~6MB) |
| 5 | Drawing Utils | Defines box drawing & HUD overlay functions |
| 6 | ⚙️ **Settings** | **Edit this** — choose video/webcam, confidence, etc. |
| 7 | Download Video | Auto-downloads a sample video if needed |
| 8 | 🚀 **Run Pipeline** | Main detection + tracking loop |
| 9 | Preview | Shows sample frames from the output video |
| 10 | Stats Chart | Bar chart of detected object classes |

---

## 🔧 Configuration

In **Step 6** of the notebook, you can customize everything:

```python
# ── INPUT ──────────────────────────────────────
USE_WEBCAM        = False              # True = live webcam, False = video file
WEBCAM_INDEX      = 0                  # Webcam device index (usually 0)
VIDEO_FILE_PATH   = "sample_video.mp4" # Path to your video file

# ── DETECTION ──────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35   # Detection confidence (0.0–1.0)
CLASSES_TO_DETECT    = None   # None = all 80 classes
                               # [0]     → people only
                               # [0, 2]  → people + cars
                               # [16]    → dogs only

# ── PROCESSING ─────────────────────────────────
RESIZE_WIDTH  = 640   # Frame width for processing (smaller = faster)
FRAME_SKIP    = 0     # Skip N frames between detections (0 = process all)
MAX_FRAMES    = 300   # Max frames to process (None = entire video)

# ── OUTPUT ─────────────────────────────────────
SAVE_OUTPUT_VIDEO  = True
OUTPUT_VIDEO_PATH  = "output_tracked.mp4"
SHOW_IN_NOTEBOOK   = True   # Preview frames inside the notebook
```

### Common Class IDs (COCO Dataset)

| ID | Class | ID | Class | ID | Class |
|---|---|---|---|---|---|
| 0 | person | 7 | truck | 16 | dog |
| 1 | bicycle | 14 | bird | 17 | cat |
| 2 | car | 15 | horse | 24 | backpack |
| 3 | motorcycle | 39 | bottle | 63 | laptop |
| 5 | bus | 41 | cup | 67 | cell phone |

---

## 📊 Output

After running the pipeline, you'll find these files in your project folder:

### `output_tracked.mp4`
The processed video with:
- 🎨 **Colored bounding boxes** — unique color per object class
- 🔢 **Tracking IDs** — persistent ID number for each object across frames
- 🏷️ **Labels** — class name + confidence score
- 📐 **Corner accents** — stylish L-shaped corners on each box
- 📟 **HUD overlay** — live FPS, frame count, and object count

### `preview.png`
A side-by-side strip of 5 sample frames from across the video.

### `detection_stats.png`
A bar chart showing how many times each object class was detected across all frames.

---

## 🐛 Common Errors & Fixes

<details>
<summary><b>❌ NameError: name 'np' is not defined</b></summary>

**Cause:** Cells were run out of order.

**Fix:** Click **⏩ Run All** (`Ctrl + Alt + R`) to run all cells from top to bottom.
</details>

<details>
<summary><b>❌ NameError: name 'YOLO' is not defined</b></summary>

**Cause:** Step 2 (imports cell) wasn't run before Step 4.

**Fix:** Run all cells in order using **Run All**.
</details>

<details>
<summary><b>❌ OSError: No such file or directory (during pip install on Windows)</b></summary>

**Cause:** Your project folder path is too long (Windows 260-char limit) or inside OneDrive.

**Fix:**
```powershell
# 1. Enable long paths (run as Administrator)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# 2. Move project to a short path
mkdir C:\od
cd C:\od
python -m venv venv
venv\Scripts\activate
pip install ultralytics opencv-python numpy filterpy scipy Pillow matplotlib lapx ipykernel jupyter
```
</details>

<details>
<summary><b>❌ source: command not found (Windows)</b></summary>

**Cause:** `source` is a Linux/Mac command.

**Fix:** Use the Windows equivalent:
```powershell
venv\Scripts\activate        # PowerShell
venv\Scripts\activate.bat    # Command Prompt (cmd.exe)
```
</details>

<details>
<summary><b>❌ Kernel "Python (object-detection)" not visible in VS Code</b></summary>

**Fix:** Point VS Code directly to the venv Python executable:

In the kernel picker → **"Python Environments..."** → **"Enter interpreter path..."** → paste:
```
C:\od\venv\Scripts\python.exe
```
</details>

<details>
<summary><b>❌ Could not open video source</b></summary>

**Cause:** Video file not found or wrong path.

**Fix:** Make sure `sample_video.mp4` exists in the same folder as the notebook. Run Step 7 to auto-download it, or manually place any `.mp4` file there and update `VIDEO_FILE_PATH` in Step 6.
</details>

---

## 💡 Customization

### Use a different YOLO model
```python
# In Step 4, change the model name:
model = YOLO('yolov8n.pt')   # nano  — fastest, least accurate (default)
model = YOLO('yolov8s.pt')   # small — good balance
model = YOLO('yolov8m.pt')   # medium — better accuracy, slower
```

### Detect only specific objects
```python
# In Step 6:
CLASSES_TO_DETECT = [0]       # People only
CLASSES_TO_DETECT = [0, 2]   # People and cars
CLASSES_TO_DETECT = [0, 1, 2, 3, 5, 7]  # All vehicle + person classes
```

### Speed up processing on CPU
```python
RESIZE_WIDTH = 416    # Smaller = faster
FRAME_SKIP   = 1      # Process every other frame
MAX_FRAMES   = 150    # Limit total frames
```

### Use your webcam instead of a video file
```python
USE_WEBCAM   = True
WEBCAM_INDEX = 0      # Try 1 or 2 if 0 doesn't work
```

---

## 🙌 Acknowledgements

| Resource | Link |
|---|---|
| **YOLOv8 by Ultralytics** | [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| **SORT Algorithm** | [github.com/abewley/sort](https://github.com/abewley/sort) |
| **OpenCV** | [opencv.org](https://opencv.org) |
| **FilterPy (Kalman Filter)** | [github.com/rlabbe/filterpy](https://github.com/rlabbe/filterpy) |
| **COCO Dataset (80 classes)** | [cocodataset.org](https://cocodataset.org) |

---

<div align="center">

Made with ❤️ as part of the **CodeAlpha Internship** Tasks

⭐ Star this repo if you found it helpful!

</div>
