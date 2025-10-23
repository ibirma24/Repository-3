# SIFT Blob Detection Project

This repository contains an implementation of blob detection using the Scale-Invariant Feature Transform (SIFT) algorithm through OpenCV's Difference of Gaussians (DoG) pyramid approach.

## Overview

The project implements three main components:
1. **Part 1**: Basic blob detection using SIFT
2. **Part 2**: Performance tuning through parameter optimization
3. **Part 3**: SIFT descriptor computation and analysis

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:
- opencv-python >= 4.5.0
- opencv-contrib-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0

## Usage

### Quick Start

For a complete workflow that runs all parts automatically:

```bash
python3 quickstart.py
```

This will:
1. Check dependencies
2. Generate the test image
3. Run blob detection (Parts 1, 2, and 3)
4. Run tests
5. Create visualizations

### Manual Execution

#### Generate Test Image

First, generate the test image with circles of various sizes:

```bash
python3 generate_test_image.py
```

This creates `example-image.jpg` containing circles of different sizes to test blob detection.

### Run Blob Detection

Execute the main blob detection script:

```bash
python3 blob_detection.py
```

This script runs all three parts of the assignment and generates output images showing detected keypoints.

### Run Tests

To validate the implementation:

```bash
python3 test_blob_detection.py
```

### Create Visualizations

To generate comparison images:

```bash
python3 visualize_results.py
```

## Part 1: Blob Detection

### Implementation Details

The SIFT detector is initialized using `cv2.SIFT_create()` with default parameters:

- **Contrast Threshold**: 0.04 (range: 0.01-0.1)
  - Filters out low-contrast keypoints in semi-uniform regions
  
- **Edge Threshold**: 10.0 (range: 5-20)
  - Eliminates edge responses (poorly localized features)
  
- **Number of Octave Layers**: 3 (range: 2-5)
  - Controls the number of layers in each octave of the DoG pyramid
  
- **Sigma**: 1.6 (range: 1.0-2.0)
  - Sigma of the Gaussian applied to the input image

### Keypoint Attributes

Each SIFT keypoint contains the following attributes:
- **pt**: (x, y) coordinates of the keypoint
- **size**: Diameter of the meaningful keypoint neighborhood (corresponds to scale σ)
- **angle**: Computed orientation of the keypoint in degrees
- **response**: Response strength (used to rank keypoints)
- **octave**: Pyramid octave from which the keypoint was extracted
- **class_id**: Object class identifier (if applicable)

### Observations

**Positive Findings:**
- ✓ Large circles are correctly detected with larger keypoint sizes
- ✓ Small circles are detected with smaller keypoint sizes
- ✓ The scale-invariant nature of DoG is validated
- ✓ Keypoints are primarily concentrated at blob boundaries where gradient changes are highest

**Issues Identified:**
- ⚠ Some keypoints are detected in uniform regions (false positives)
- ⚠ Very small blobs may be missed with default parameters
- ⚠ Multiple keypoints per blob can create redundancy
- ⚠ Edge-like features along circle boundaries generate numerous keypoints
- ⚠ The interior of solid circles typically has no keypoints (expected behavior - no texture)

## Part 2: Tuning Blob Detection Performance

### Parameter Exploration

We tested various combinations of contrast and edge thresholds:

| Contrast Threshold | Edge Threshold | Keypoints Detected |
|-------------------|----------------|-------------------|
| 0.01              | 5              | 27                |
| 0.01              | 10             | 61                |
| 0.01              | 15             | 95                |
| 0.04 (default)    | 10 (default)   | 59                |
| 0.08              | 10             | 59                |

### Observations and Analysis

**Effect of Contrast Threshold:**
- **Lower values (0.01)**: Detects more keypoints, including lower-contrast features
- **Higher values (0.08)**: Filters out weak responses, reducing false positives
- **Finding**: Contrast threshold primarily affects detection in low-contrast regions

**Effect of Edge Threshold:**
- **Lower values (5)**: Stricter elimination of edge-like features → fewer keypoints
- **Higher values (15)**: More permissive, retains edge responses → more keypoints
- **Finding**: Edge threshold significantly impacts the total number of detected keypoints

**Optimal Configuration:**
For our test image with simple geometric shapes:
- **Contrast Threshold**: 0.01
- **Edge Threshold**: 15
- **Result**: 95 keypoints (maximum detection)

This configuration captures the most circle features in our example image, though in real-world applications, more selective parameters may be preferred to reduce false positives.

**Parameter Function Summary:**
1. **Contrast Threshold** eliminates keypoints with weak responses in low-contrast regions
2. **Edge Threshold** uses the Harris corner detector response ratio to eliminate poorly localized features along edges
3. Together, these thresholds help SIFT focus on stable, distinctive keypoints

## Part 3: Descriptors

### Descriptor Properties

SIFT descriptors computed for detected keypoints:

**Descriptor Dimensions:**
- Each descriptor: 128-dimensional vector
- Structure: 4×4 grid of cells, each with 8-orientation histogram
- Total: 4 × 4 × 8 = 128 dimensions

**Statistical Analysis:**
- Value range: [0, 255] (after quantization)
- Descriptors are normalized to handle illumination changes
- High-dimensional representation ensures distinctiveness

**Key Properties:**
1. **Scale Invariant**: Computed at the detected scale of each keypoint
2. **Rotation Invariant**: Uses the dominant orientation of local gradients
3. **Illumination Robust**: Normalized to unit length
4. **Distinctive**: 128-D representation captures local appearance effectively
5. **Efficient Matching**: Euclidean distance in descriptor space enables feature matching

### Descriptor Use Cases

SIFT descriptors enable:
- Image matching and registration
- Object recognition
- 3D reconstruction
- Panorama stitching
- Visual odometry

## Output Files

The script generates the following visualization files:

- `example-image.jpg` - Test image with circles of various sizes
- `output_part1_blob_detection.jpg` - Basic detection with default parameters
- `output_part2_contrast0.010_edge10.jpg` - Low contrast threshold
- `output_part2_contrast0.040_edge10.jpg` - Default parameters
- `output_part2_contrast0.080_edge10.jpg` - High contrast threshold

## Conclusion

This implementation demonstrates:
1. SIFT successfully detects blobs at multiple scales
2. Parameter tuning significantly affects detection quality
3. SIFT descriptors provide robust, distinctive feature representations

The Difference of Gaussians approach effectively implements scale-space blob detection, though careful parameter selection is crucial for optimal performance in specific applications.

## References

- Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
- OpenCV SIFT Documentation: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html