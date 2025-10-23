"""
SIFT Blob Detection Implementation
This script implements blob detection using OpenCV's SIFT (Scale-Invariant Feature Transform) algorithm.

Parts implemented:
1. Blob Detection - Initialize SIFT, detect keypoints, visualize results
2. Tuning Performance - Adjust thresholds to optimize detection
3. Descriptors - Compute and analyze SIFT descriptors
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def print_sift_parameters(sift):
    """
    Print the tunable parameters of the SIFT detector.
    """
    print("\n" + "="*70)
    print("SIFT DETECTOR PARAMETERS")
    print("="*70)
    
    # Get parameters using getDefaultName and descriptor methods
    print(f"Contrast Threshold: {sift.getContrastThreshold()}")
    print(f"  - Default: 0.04")
    print(f"  - Range: Typically 0.01 to 0.1")
    print(f"  - Purpose: Filters out low-contrast keypoints in semi-uniform regions")
    print()
    
    print(f"Edge Threshold: {sift.getEdgeThreshold()}")
    print(f"  - Default: 10")
    print(f"  - Range: Typically 5 to 20")
    print(f"  - Purpose: Eliminates edge responses (poorly localized features)")
    print()
    
    print(f"Number of Octave Layers: {sift.getNOctaveLayers()}")
    print(f"  - Default: 3")
    print(f"  - Range: 2 to 5")
    print(f"  - Purpose: Number of layers in each octave of the DoG pyramid")
    print()
    
    print(f"Sigma: {sift.getSigma()}")
    print(f"  - Default: 1.6")
    print(f"  - Range: 1.0 to 2.0")
    print(f"  - Purpose: Sigma of the Gaussian applied to the input image")
    print("="*70 + "\n")


def analyze_keypoints(keypoints):
    """
    Analyze and print information about detected keypoints.
    """
    print("\n" + "="*70)
    print("KEYPOINT ANALYSIS")
    print("="*70)
    print(f"Total keypoints detected: {len(keypoints)}")
    
    if len(keypoints) > 0:
        # Extract keypoint properties
        sizes = [kp.size for kp in keypoints]
        angles = [kp.angle for kp in keypoints]
        responses = [kp.response for kp in keypoints]
        
        print(f"\nSize (scale) statistics:")
        print(f"  - Min: {min(sizes):.2f}")
        print(f"  - Max: {max(sizes):.2f}")
        print(f"  - Mean: {np.mean(sizes):.2f}")
        print(f"  - Median: {np.median(sizes):.2f}")
        
        print(f"\nResponse (strength) statistics:")
        print(f"  - Min: {min(responses):.4f}")
        print(f"  - Max: {max(responses):.4f}")
        print(f"  - Mean: {np.mean(responses):.4f}")
        
        print(f"\nKeypoint attributes:")
        print(f"  - pt: (x, y) coordinates of the keypoint")
        print(f"  - size: diameter of the meaningful keypoint neighborhood")
        print(f"  - angle: computed orientation of the keypoint (-1 if not applicable)")
        print(f"  - response: response strength (used to rank keypoints)")
        print(f"  - octave: octave (pyramid layer) from which the keypoint was extracted")
        print(f"  - class_id: object class (if applicable)")
        
        # Sample keypoint details
        print(f"\nSample keypoint details (first 3):")
        for i, kp in enumerate(keypoints[:3]):
            print(f"  Keypoint {i+1}:")
            print(f"    - Position: ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})")
            print(f"    - Size: {kp.size:.2f}")
            print(f"    - Angle: {kp.angle:.2f}°")
            print(f"    - Response: {kp.response:.4f}")
            print(f"    - Octave: {kp.octave}")
    
    print("="*70 + "\n")


def part1_blob_detection(image_path):
    """
    Part 1: Basic blob detection using SIFT.
    """
    print("\n" + "#"*70)
    print("# PART 1: BLOB DETECTION")
    print("#"*70)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Print tunable parameters
    print_sift_parameters(sift)
    
    # Detect keypoints
    keypoints = sift.detect(gray, None)
    
    # Analyze keypoints
    analyze_keypoints(keypoints)
    
    # Visualize keypoints with size
    output_image = cv2.drawKeypoints(
        image, 
        keypoints, 
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Save output
    output_path = 'output_part1_blob_detection.jpg'
    cv2.imwrite(output_path, output_image)
    print(f"Part 1 output saved as: {output_path}")
    
    return sift, keypoints


def part2_tuning_performance(image_path):
    """
    Part 2: Tune blob detection performance by adjusting thresholds.
    """
    print("\n" + "#"*70)
    print("# PART 2: TUNING BLOB DETECTION PERFORMANCE")
    print("#"*70)
    
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Test different contrast thresholds
    contrast_thresholds = [0.01, 0.04, 0.08]
    edge_thresholds = [5, 10, 15]
    
    results = []
    
    print("\nTesting different parameter combinations:\n")
    
    for contrast_thresh in contrast_thresholds:
        for edge_thresh in edge_thresholds:
            # Create SIFT detector with custom parameters
            sift = cv2.SIFT_create(
                contrastThreshold=contrast_thresh,
                edgeThreshold=edge_thresh
            )
            
            # Detect keypoints
            keypoints = sift.detect(gray, None)
            
            # Store results
            results.append({
                'contrast_threshold': contrast_thresh,
                'edge_threshold': edge_thresh,
                'num_keypoints': len(keypoints),
                'keypoints': keypoints
            })
            
            print(f"Contrast: {contrast_thresh:.3f}, Edge: {edge_thresh:2d} -> "
                  f"Keypoints: {len(keypoints)}")
    
    # Visualize best configurations
    print("\nGenerating visualizations for different configurations...\n")
    
    # Select a few interesting configurations to visualize
    configs_to_visualize = [
        (0.01, 10),  # Low contrast threshold (more keypoints)
        (0.04, 10),  # Default parameters
        (0.08, 10),  # High contrast threshold (fewer keypoints)
    ]
    
    for contrast_thresh, edge_thresh in configs_to_visualize:
        sift = cv2.SIFT_create(
            contrastThreshold=contrast_thresh,
            edgeThreshold=edge_thresh
        )
        keypoints = sift.detect(gray, None)
        
        output_image = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        output_path = f'output_part2_contrast{contrast_thresh:.3f}_edge{edge_thresh}.jpg'
        cv2.imwrite(output_path, output_image)
        print(f"Saved: {output_path} ({len(keypoints)} keypoints)")
    
    # Find optimal configuration (most keypoints for our test image)
    best_result = max(results, key=lambda x: x['num_keypoints'])
    print(f"\nOptimal configuration for maximum keypoints:")
    print(f"  - Contrast Threshold: {best_result['contrast_threshold']:.3f}")
    print(f"  - Edge Threshold: {best_result['edge_threshold']}")
    print(f"  - Keypoints detected: {best_result['num_keypoints']}")
    
    return best_result


def part3_descriptors(image_path, keypoints):
    """
    Part 3: Compute and analyze SIFT descriptors.
    """
    print("\n" + "#"*70)
    print("# PART 3: DESCRIPTORS")
    print("#"*70)
    
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector with optimal parameters
    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=10)
    
    # Compute descriptors
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    print("\n" + "="*70)
    print("DESCRIPTOR ANALYSIS")
    print("="*70)
    
    if descriptors is not None:
        print(f"Number of keypoints with descriptors: {len(keypoints)}")
        print(f"Descriptor shape: {descriptors.shape}")
        print(f"  - Each descriptor is a {descriptors.shape[1]}-dimensional vector")
        print(f"  - Standard SIFT descriptor: 128 dimensions (4x4 grid, 8 orientations)")
        
        print(f"\nDescriptor statistics:")
        print(f"  - Min value: {descriptors.min():.4f}")
        print(f"  - Max value: {descriptors.max():.4f}")
        print(f"  - Mean value: {descriptors.mean():.4f}")
        print(f"  - Std dev: {descriptors.std():.4f}")
        
        print(f"\nSample descriptor (first keypoint):")
        print(f"  {descriptors[0][:16]}...")  # Show first 16 values
        
        print(f"\nDescriptor properties:")
        print(f"  - Rotation invariant: Uses dominant orientation of gradient")
        print(f"  - Scale invariant: Computed at the keypoint's detected scale")
        print(f"  - Robust to illumination: Normalized to unit length")
        print(f"  - Distinctive: High-dimensional representation of local appearance")
    else:
        print("No descriptors computed!")
    
    print("="*70 + "\n")
    
    return keypoints, descriptors


def main():
    """
    Main function to run all three parts of the blob detection assignment.
    """
    image_path = 'example-image.jpg'
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        print("Please ensure the image file exists in the current directory.")
        return
    
    print("\n" + "="*70)
    print("SIFT BLOB DETECTION - Computer Vision Assignment")
    print("="*70)
    
    # Part 1: Blob Detection
    sift, keypoints = part1_blob_detection(image_path)
    
    # Part 2: Tuning Performance
    best_result = part2_tuning_performance(image_path)
    
    # Part 3: Descriptors
    keypoints_with_desc, descriptors = part3_descriptors(
        image_path, 
        best_result['keypoints']
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated output files:")
    print("  - output_part1_blob_detection.jpg")
    print("  - output_part2_contrast0.010_edge10.jpg")
    print("  - output_part2_contrast0.040_edge10.jpg")
    print("  - output_part2_contrast0.080_edge10.jpg")
    print("\nPlease review these images and update README.md with your observations.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
