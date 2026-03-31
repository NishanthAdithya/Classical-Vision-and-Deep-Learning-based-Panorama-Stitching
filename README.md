

# MyAutoPano — Classical and Deep Learning based Panorama Stitching From Scratch

This project implements a complete panorama stitching pipeline using classical computer vision techniques and deep learning architectures.
Given a sequence of images of about 30-50 percent overlap, the system automatically detects features, matches them across images, estimates homographies using RANSAC, and blends the images into a seamless panorama. In Phase 2, a supervised and unsupervised neural network is trained on synthetic data to directly estimate the homography.

---

## Folder Structure



```
├── Phase1 
├── Phase2
│   ├── Code
│   │   ├── Misc
│   │   ├── Network
│   │   ├── Results
│   │   ├── Test.py
│   │   ├── Train.py
│   │   ├── TxtFiles
│   │   └── Wrapper.py 
├── README.md
└── Report.pdf
```

---

## Requirements

- Python 3.8+
- numpy
- OpenCV
- Matplotlib
- SciPy
- networkx
- kornia

Install dependencies:

```bash
pip install numpy opencv-python matplotlib scipy networkx kornia

## How To Run - Phase 1

### Step 1 — Place Input Images

Copy all overlapping images into:

### Step 2 — Navigate to Code Directory

From the project root:

```bash
cd Phase1/Code

### Step 3 — Run the Panorama Stitching Pipeline

```bash
python Wrapper.py

### Step 4 — View Outputs

All intermediate and final results will be saved in:
```bash
Phase1/output_visualizations/

## How To Run - Phase 2

### Step 1 — Training

Choose supervised or unsupervised model type with Sup and Unsup.

From the project root:

```bash
python Phase2/Code/Train.py

### Step 3 — Testing

```bash
python Phase2/Code/Test.py

### Step 4 — Panorama stitching of video

```bash
python Phase2/Code/Wrapper.py

