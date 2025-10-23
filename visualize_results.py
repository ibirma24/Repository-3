"""
Display all generated output images for visual inspection.
"""
import cv2
import matplotlib.pyplot as plt
import os


def display_images():
    """Display all generated images."""
    images = {
        'Original Image': 'example-image.jpg',
        'Part 1: Default Detection': 'output_part1_blob_detection.jpg',
        'Part 2: Low Contrast (0.01)': 'output_part2_contrast0.010_edge10.jpg',
        'Part 2: Default (0.04)': 'output_part2_contrast0.040_edge10.jpg',
        'Part 2: High Contrast (0.08)': 'output_part2_contrast0.080_edge10.jpg',
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SIFT Blob Detection Results', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    idx = 0
    for title, filename in images.items():
        if os.path.exists(filename):
            image = cv2.imread(filename)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(image_rgb)
            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')
            idx += 1
    
    # Hide unused subplot
    if idx < len(axes):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
    print("✓ Comparison saved as: comparison_results.png")
    
    # Also create individual keypoint count visualization
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    configs = ['Low\n(0.01, 10)', 'Default\n(0.04, 10)', 'High\n(0.08, 10)']
    keypoint_counts = [61, 59, 59]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(configs, keypoint_counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Configuration (Contrast, Edge)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Keypoints Detected', fontsize=12, fontweight='bold')
    ax.set_title('Effect of Contrast Threshold on Keypoint Detection', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, keypoint_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('keypoint_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Keypoint comparison saved as: keypoint_comparison.png")


if __name__ == "__main__":
    display_images()
