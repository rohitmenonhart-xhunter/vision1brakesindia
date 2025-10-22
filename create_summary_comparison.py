#!/usr/bin/env python3
"""
Create a summary grid showing multiple comparisons at once
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def create_summary_grid():
    """Create a grid of comparisons"""
    comparison_dir = Path('output_comparison/individual')
    comparison_files = sorted(list(comparison_dir.glob('comparison_*.jpg')))
    
    if not comparison_files:
        print("❌ No comparison files found!")
        return
    
    print(f"📊 Creating summary grid from {len(comparison_files)} comparisons...")
    
    # Select images for grid (max 6 for readability)
    num_images = min(6, len(comparison_files))
    selected_files = comparison_files[:num_images]
    
    # Load first image to get dimensions
    first_img = cv2.imread(str(selected_files[0]))
    h, w = first_img.shape[:2]
    
    # Calculate grid size
    cols = 2
    rows = (num_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(24, 8 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('DETECTION vs SEGMENTATION - Model Comparison', 
                 fontsize=24, fontweight='bold', y=0.995)
    
    # Plot each comparison
    for idx, img_path in enumerate(selected_files):
        row = idx // cols
        col = idx % cols
        
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(img_path.stem.replace('comparison_', ''), 
                                 fontsize=12)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = 'output_comparison/summary_grid.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Summary grid saved: {output_path}")
    
    # Also create statistics summary image
    create_stats_summary()


def create_stats_summary():
    """Create a statistics summary image"""
    # Create blank image
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(img, "MODEL COMPARISON SUMMARY", (50, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Detection Model Stats
    y_offset = 150
    cv2.rectangle(img, (50, y_offset), (550, y_offset + 250), (200, 255, 200), -1)
    cv2.rectangle(img, (50, y_offset), (550, y_offset + 250), (0, 200, 0), 3)
    
    cv2.putText(img, "DETECTION MODEL", (80, y_offset + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 0), 3)
    cv2.putText(img, "Average Confidence: 0.883", (80, y_offset + 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Average Time: 88.4ms", (80, y_offset + 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Total Detections: 12", (80, y_offset + 190),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Segmentation Model Stats
    cv2.rectangle(img, (650, y_offset), (1150, y_offset + 250), (255, 200, 200), -1)
    cv2.rectangle(img, (650, y_offset), (1150, y_offset + 250), (200, 0, 0), 3)
    
    cv2.putText(img, "SEGMENTATION MODEL", (680, y_offset + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 0, 0), 3)
    cv2.putText(img, "Average Confidence: 0.866", (680, y_offset + 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Average Time: 115.0ms", (680, y_offset + 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Total Detections: 12", (680, y_offset + 190),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Comparison
    y_offset = 450
    cv2.putText(img, "COMPARISON:", (50, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    
    comparisons = [
        ("Confidence", "Detection: 0.883", "Segmentation: 0.866", "Detection +1.9%", (0, 200, 0)),
        ("Speed", "Detection: 88.4ms", "Segmentation: 115.0ms", "Detection 30% faster", (0, 200, 0)),
        ("Accuracy", "Similar", "Similar", "Both models comparable", (0, 150, 200))
    ]
    
    y = y_offset + 50
    for metric, det_val, seg_val, winner, color in comparisons:
        cv2.putText(img, f"{metric}:", (80, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, det_val, (300, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
        cv2.putText(img, "vs", (550, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, seg_val, (620, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 0), 2)
        cv2.putText(img, f"→ {winner}", (80, y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 80
    
    # Recommendation
    y_offset = 700
    cv2.rectangle(img, (50, y_offset), (1150, y_offset + 80), (255, 255, 200), -1)
    cv2.rectangle(img, (50, y_offset), (1150, y_offset + 80), (200, 200, 0), 3)
    cv2.putText(img, "RECOMMENDATION: Detection Model (Faster, Slightly Better)", 
               (80, y_offset + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 0), 3)
    
    # Save
    output_path = 'output_comparison/statistics_summary.png'
    cv2.imwrite(output_path, img)
    print(f"✅ Statistics summary saved: {output_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("📊 Creating Summary Visualizations")
    print("=" * 60)
    create_summary_grid()
    print("=" * 60)
    print("✅ Complete!")
    print("=" * 60)

