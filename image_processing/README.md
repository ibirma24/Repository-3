# SIFT Feature Analysis

## Overview
This project implements SIFT (Scale-Invariant Feature Transform) feature analysis for computer vision applications. SIFT is a computer vision algorithm used to detect and describe local features in images.

## Project Structure
```
image_processing/
├── README.md              # This file
├── sift_analysis.py       # Main SIFT analysis implementation
├── requirements.txt       # Python dependencies
└── images/
    └── example-image.jpg  # Sample image for testing
```

## Requirements
- Python 3.7+
- OpenCV (opencv-python)
- NumPy
- Matplotlib (for visualization)

## Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the SIFT feature analysis on an image:
```bash
python sift_analysis.py images/example-image.jpg
```

This will:
1. Load the specified image
2. Detect SIFT keypoints
3. Compute SIFT descriptors
4. Display the image with keypoints marked
5. Save the output image with keypoints

## Features
- **Keypoint Detection**: Identifies distinctive keypoints in the image
- **Feature Description**: Computes 128-dimensional SIFT descriptors for each keypoint
- **Visualization**: Displays detected keypoints overlaid on the original image
- **Output Generation**: Saves processed images with detected features

## SIFT Algorithm
SIFT features are invariant to:
- Image scale
- Rotation
- Illumination changes
- Minor viewpoint changes

The algorithm consists of four main steps:
1. Scale-space extrema detection
2. Keypoint localization
3. Orientation assignment
4. Keypoint descriptor generation

## Output
The program outputs:
- Number of keypoints detected
- Visualization of keypoints on the image
- Saved image with keypoints marked

## About
CSC 391: Special Topics - Computer Vision
Assignment: Repository 3 - SIFT Feature Analysis
Due: Tuesday Oct 28
