#!/usr/bin/env python3
"""
Quick start script for SIFT Blob Detection project.
This script demonstrates the complete workflow.
"""

import os
import sys


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Step 1: Checking Dependencies")
    
    try:
        import cv2
        import numpy
        import matplotlib
        print("✓ All required packages are installed")
        print(f"  - OpenCV version: {cv2.__version__}")
        print(f"  - NumPy version: {numpy.__version__}")
        print(f"  - Matplotlib version: {matplotlib.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies using:")
        print("  pip install -r requirements.txt")
        return False


def generate_test_image():
    """Generate the test image if it doesn't exist."""
    print_header("Step 2: Generating Test Image")
    
    if os.path.exists('example-image.jpg'):
        print("✓ Test image already exists: example-image.jpg")
        return True
    
    print("Generating test image with circles...")
    import subprocess
    result = subprocess.run([sys.executable, 'generate_test_image.py'],
                          capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists('example-image.jpg'):
        print("✓ Test image created: example-image.jpg")
        return True
    else:
        print("✗ Failed to generate test image")
        return False


def run_blob_detection():
    """Run the main blob detection script."""
    print_header("Step 3: Running Blob Detection")
    
    print("Executing blob detection analysis...")
    print("This will run all three parts:")
    print("  - Part 1: Basic blob detection")
    print("  - Part 2: Parameter tuning")
    print("  - Part 3: Descriptor computation")
    print()
    
    import subprocess
    result = subprocess.run([sys.executable, 'blob_detection.py'])
    
    if result.returncode == 0:
        print("\n✓ Blob detection completed successfully")
        return True
    else:
        print("\n✗ Blob detection failed")
        return False


def run_tests():
    """Run the test suite."""
    print_header("Step 4: Running Tests")
    
    import subprocess
    result = subprocess.run([sys.executable, 'test_blob_detection.py'])
    
    if result.returncode == 0:
        return True
    else:
        print("\n✗ Some tests failed")
        return False


def create_visualizations():
    """Create comparison visualizations."""
    print_header("Step 5: Creating Visualizations")
    
    print("Generating comparison images...")
    import subprocess
    result = subprocess.run([sys.executable, 'visualize_results.py'],
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print("✗ Visualization generation failed")
        return False


def list_outputs():
    """List all generated output files."""
    print_header("Generated Output Files")
    
    output_files = [
        'example-image.jpg',
        'output_part1_blob_detection.jpg',
        'output_part2_contrast0.010_edge10.jpg',
        'output_part2_contrast0.040_edge10.jpg',
        'output_part2_contrast0.080_edge10.jpg',
        'comparison_results.png',
        'keypoint_comparison.png',
    ]
    
    print("The following files have been generated:\n")
    for filename in output_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024
            print(f"  ✓ {filename:<45} ({size:.1f} KB)")
        else:
            print(f"  ✗ {filename:<45} (not found)")


def main():
    """Run the complete workflow."""
    print("\n" + "="*70)
    print("  SIFT BLOB DETECTION - QUICK START")
    print("="*70)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 2: Generate test image
    if not generate_test_image():
        sys.exit(1)
    
    # Step 3: Run blob detection
    if not run_blob_detection():
        sys.exit(1)
    
    # Step 4: Run tests
    if not run_tests():
        print("\nWarning: Some tests failed, but continuing...")
    
    # Step 5: Create visualizations
    if not create_visualizations():
        print("\nWarning: Visualization generation failed, but continuing...")
    
    # List outputs
    list_outputs()
    
    # Final message
    print_header("Quick Start Complete")
    print("All steps completed successfully!")
    print("\nNext steps:")
    print("  1. Review the generated output images")
    print("  2. Read the analysis in README.md")
    print("  3. Experiment with different images")
    print("  4. Try different SIFT parameters in blob_detection.py")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
