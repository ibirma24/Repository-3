#!/usr/bin/env python3
"""
SIFT Feature Analysis
CSC 391: Special Topics - Computer Vision

This script performs SIFT (Scale-Invariant Feature Transform) feature detection
and analysis on images.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        The loaded image in BGR format, or None if loading fails
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from '{image_path}'.")
        return None
    
    return image


def detect_sift_features(image):
    """
    Detect SIFT keypoints and compute descriptors.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors


def draw_keypoints(image, keypoints):
    """
    Draw detected keypoints on the image.
    
    Args:
        image: Input image in BGR format
        keypoints: List of detected keypoints
        
    Returns:
        Image with keypoints drawn
    """
    # Draw keypoints with rich information (size and orientation)
    output_image = cv2.drawKeypoints(
        image, 
        keypoints, 
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    return output_image


def display_results(image, keypoints, descriptors):
    """
    Display the results of SIFT feature detection.
    
    Args:
        image: Original image
        keypoints: Detected keypoints
        descriptors: Computed descriptors
    """
    # Draw keypoints on image
    output_image = draw_keypoints(image, keypoints)
    
    # Display using matplotlib
    plt.figure(figsize=(12, 8))
    
    # Convert BGR to RGB for matplotlib
    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(output_rgb)
    plt.title(f'SIFT Features Detected: {len(keypoints)} keypoints')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_output(image, keypoints, output_path):
    """
    Save the image with detected keypoints.
    
    Args:
        image: Original image
        keypoints: Detected keypoints
        output_path: Path to save the output image
    """
    output_image = draw_keypoints(image, keypoints)
    cv2.imwrite(output_path, output_image)
    print(f"Output saved to: {output_path}")


def print_keypoint_statistics(keypoints, descriptors):
    """
    Print statistics about detected keypoints.
    
    Args:
        keypoints: List of detected keypoints
        descriptors: Array of descriptors
    """
    print("\n=== SIFT Feature Analysis Results ===")
    print(f"Number of keypoints detected: {len(keypoints)}")
    
    if descriptors is not None:
        print(f"Descriptor shape: {descriptors.shape}")
        print(f"Descriptor dimensions: {descriptors.shape[1]} (standard SIFT)")
    
    if len(keypoints) > 0:
        # Calculate average keypoint size
        sizes = [kp.size for kp in keypoints]
        avg_size = np.mean(sizes)
        print(f"Average keypoint size: {avg_size:.2f}")
        
        # Get response values
        responses = [kp.response for kp in keypoints]
        print(f"Response range: [{min(responses):.4f}, {max(responses):.4f}]")
    
    print("=" * 38)


def main():
    """Main function to run SIFT feature analysis."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python sift_analysis.py <image_path>")
        print("Example: python sift_analysis.py images/example-image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load the image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    if image is None:
        sys.exit(1)
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Detect SIFT features
    print("Detecting SIFT features...")
    keypoints, descriptors = detect_sift_features(image)
    
    # Print statistics
    print_keypoint_statistics(keypoints, descriptors)
    
    # Save output image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base_name}_sift_features.jpg"
    save_output(image, keypoints, output_path)
    
    # Display results
    print("\nDisplaying results...")
    display_results(image, keypoints, descriptors)


if __name__ == "__main__":
    main()
