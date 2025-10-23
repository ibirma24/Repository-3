"""
Script to generate a test image with circles of various sizes for blob detection testing.
"""
import cv2
import numpy as np

# Create a white image
image = np.ones((600, 800, 3), dtype=np.uint8) * 255

# Draw circles of various sizes (blobs)
circles = [
    ((150, 150), 40, (0, 0, 0)),      # Small black circle
    ((400, 150), 60, (100, 100, 100)), # Medium gray circle
    ((650, 150), 80, (50, 50, 50)),    # Large dark gray circle
    ((200, 350), 50, (0, 0, 0)),       # Medium black circle
    ((450, 350), 70, (80, 80, 80)),    # Large gray circle
    ((150, 500), 30, (0, 0, 0)),       # Small black circle
    ((550, 450), 90, (60, 60, 60)),    # Extra large gray circle
]

for center, radius, color in circles:
    cv2.circle(image, center, radius, color, -1)

# Save the image
cv2.imwrite('example-image.jpg', image)
print("Test image 'example-image.jpg' created successfully!")
print(f"Image shape: {image.shape}")
