"""
Test script to validate the blob detection implementation.
"""
import cv2
import os
import sys


def test_image_exists():
    """Test that the example image exists."""
    assert os.path.exists('example-image.jpg'), "example-image.jpg not found"
    print("✓ Test passed: example-image.jpg exists")


def test_sift_initialization():
    """Test SIFT detector initialization."""
    sift = cv2.SIFT_create()
    assert sift is not None, "SIFT detector initialization failed"
    
    # Verify parameters
    assert sift.getContrastThreshold() == 0.04, "Default contrast threshold incorrect"
    assert sift.getEdgeThreshold() == 10.0, "Default edge threshold incorrect"
    assert sift.getNOctaveLayers() == 3, "Default octave layers incorrect"
    assert sift.getSigma() == 1.6, "Default sigma incorrect"
    
    print("✓ Test passed: SIFT initialization and default parameters")


def test_keypoint_detection():
    """Test keypoint detection on the example image."""
    image = cv2.imread('example-image.jpg')
    assert image is not None, "Failed to load example-image.jpg"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    
    assert len(keypoints) > 0, "No keypoints detected"
    print(f"✓ Test passed: {len(keypoints)} keypoints detected")


def test_keypoint_attributes():
    """Test that keypoints have expected attributes."""
    image = cv2.imread('example-image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    
    kp = keypoints[0]
    assert hasattr(kp, 'pt'), "Keypoint missing 'pt' attribute"
    assert hasattr(kp, 'size'), "Keypoint missing 'size' attribute"
    assert hasattr(kp, 'angle'), "Keypoint missing 'angle' attribute"
    assert hasattr(kp, 'response'), "Keypoint missing 'response' attribute"
    assert hasattr(kp, 'octave'), "Keypoint missing 'octave' attribute"
    
    print("✓ Test passed: Keypoint attributes validated")


def test_parameter_tuning():
    """Test SIFT with different parameters."""
    image = cv2.imread('example-image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Test with low contrast threshold
    sift_low = cv2.SIFT_create(contrastThreshold=0.01)
    keypoints_low = sift_low.detect(gray, None)
    
    # Test with high contrast threshold
    sift_high = cv2.SIFT_create(contrastThreshold=0.08)
    keypoints_high = sift_high.detect(gray, None)
    
    # Lower threshold should detect more or equal keypoints
    assert len(keypoints_low) >= len(keypoints_high), \
        "Low threshold should detect more keypoints"
    
    print(f"✓ Test passed: Parameter tuning (low: {len(keypoints_low)}, high: {len(keypoints_high)})")


def test_descriptor_computation():
    """Test descriptor computation."""
    image = cv2.imread('example-image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    assert descriptors is not None, "Descriptors not computed"
    assert descriptors.shape[0] == len(keypoints), "Descriptor count mismatch"
    assert descriptors.shape[1] == 128, "Descriptor dimension should be 128"
    
    print(f"✓ Test passed: {descriptors.shape[0]} descriptors computed (128-D each)")


def test_visualization():
    """Test keypoint visualization."""
    image = cv2.imread('example-image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    
    output_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    assert output_image is not None, "Visualization failed"
    assert output_image.shape == image.shape, "Output shape mismatch"
    
    print("✓ Test passed: Keypoint visualization")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running Blob Detection Tests")
    print("="*70 + "\n")
    
    tests = [
        test_image_exists,
        test_sift_initialization,
        test_keypoint_detection,
        test_keypoint_attributes,
        test_parameter_tuning,
        test_descriptor_computation,
        test_visualization,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ Test failed: {test.__name__} - {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {test.__name__} - {e}")
            failed += 1
    
    print("\n" + "="*70)
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failed} test(s) failed!")
        sys.exit(1)
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
