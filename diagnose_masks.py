#!/usr/bin/env python3
"""
Mask Diagnostic Tool
Analyzes segmentation mask quality and visualizes issues
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


def diagnose_segmentation(image_path, model_path):
    """
    Detailed mask analysis
    """
    print(f"\n{'='*60}")
    print(f"🔍 MASK DIAGNOSTIC: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Load model and image
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    
    # Run segmentation
    results = model(image_path, verbose=False)
    
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
            print(f"\n✅ Found {len(result.masks)} mask(s)")
            
            for idx, (mask, box) in enumerate(zip(result.masks, result.boxes)):
                print(f"\n--- Mask {idx + 1} ---")
                
                # Get class info
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                
                print(f"Class: {cls_name}")
                print(f"Confidence: {conf:.3f}")
                
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                print(f"BBox: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"BBox Size: {x2-x1} x {y2-y1} pixels")
                
                # Get mask
                mask_array = mask.data[0].cpu().numpy()
                print(f"\nOriginal mask shape: {mask_array.shape}")
                print(f"Original mask range: {mask_array.min():.3f} to {mask_array.max():.3f}")
                
                # Resize to image size
                mask_resized = cv2.resize(mask_array, (img.shape[1], img.shape[0]))
                print(f"Resized mask shape: {mask_resized.shape}")
                
                # Different thresholds
                thresholds = [0.3, 0.5, 0.7, 0.9]
                print(f"\nMask pixels at different thresholds:")
                for thresh in thresholds:
                    mask_bool = mask_resized > thresh
                    pixel_count = mask_bool.sum()
                    percentage = (pixel_count / mask_bool.size) * 100
                    print(f"  Threshold {thresh}: {pixel_count} pixels ({percentage:.2f}% of image)")
                
                # Analyze mask shape
                mask_bool_05 = mask_resized > 0.5
                
                # Find contours
                contours, _ = cv2.findContours(mask_bool_05.astype(np.uint8), 
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                print(f"\nMask analysis (threshold 0.5):")
                print(f"  Contours found: {len(contours)}")
                if contours:
                    main_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(main_contour)
                    perimeter = cv2.arcLength(main_contour, True)
                    print(f"  Main contour area: {area:.0f} pixels")
                    print(f"  Main contour perimeter: {perimeter:.1f} pixels")
                    
                    # Approximate polygon
                    epsilon = 0.02 * perimeter
                    approx = cv2.approxPolyDP(main_contour, epsilon, True)
                    print(f"  Polygon vertices (epsilon=0.02): {len(approx)}")
                    
                    # Check if triangle-like
                    if len(approx) == 3:
                        print(f"  ✅ Shape: TRIANGLE (3 vertices)")
                    elif len(approx) == 4:
                        print(f"  ⚠️  Shape: QUADRILATERAL (4 vertices)")
                    else:
                        print(f"  ⚠️  Shape: POLYGON ({len(approx)} vertices)")
                
                # Create visualization
                create_mask_visualization(img, mask_resized, mask_bool_05, 
                                        box.xyxy[0].cpu().numpy(), 
                                        cls_name, conf, idx)
        else:
            print("\n❌ No masks found!")
            if result.boxes is not None and len(result.boxes) > 0:
                print("   (Model has boxes but no segmentation masks)")


def create_mask_visualization(img, mask_continuous, mask_binary, bbox, cls_name, conf, idx):
    """Create detailed mask visualization"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Mask Analysis: {cls_name} (Confidence: {conf:.3f})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Original image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Draw bbox
    x1, y1, x2, y2 = map(int, bbox)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                         fill=False, edgecolor='red', linewidth=2)
    axes[0, 0].add_patch(rect)
    
    # 2. Continuous mask (heatmap)
    im = axes[0, 1].imshow(mask_continuous, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('Mask Confidence (Continuous)')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Binary mask (threshold 0.5)
    axes[0, 2].imshow(mask_binary, cmap='gray')
    axes[0, 2].set_title('Binary Mask (Threshold 0.5)')
    axes[0, 2].axis('off')
    
    # 4. Mask overlay on image
    img_overlay = img_rgb.copy()
    overlay_colored = np.zeros_like(img_rgb)
    overlay_colored[mask_binary] = [255, 0, 0]  # Red
    img_overlay = cv2.addWeighted(img_rgb, 0.6, overlay_colored, 0.4, 0)
    axes[1, 0].imshow(img_overlay)
    axes[1, 0].set_title('Mask Overlay (Red)')
    axes[1, 0].axis('off')
    
    # 5. Mask contour
    img_contour = img_rgb.copy()
    contours, _ = cv2.findContours(mask_binary.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 3)
    axes[1, 1].imshow(img_contour)
    axes[1, 1].set_title(f'Contour (Green) - {len(contours)} contour(s)')
    axes[1, 1].axis('off')
    
    # 6. Polygon approximation
    img_poly = img_rgb.copy()
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(main_contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        cv2.drawContours(img_poly, [approx], -1, (255, 255, 0), 3)
        
        # Draw vertices
        for point in approx:
            cv2.circle(img_poly, tuple(point[0]), 8, (0, 0, 255), -1)
        
        axes[1, 2].imshow(img_poly)
        axes[1, 2].set_title(f'Polygon Approximation ({len(approx)} vertices)')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = f'mask_diagnostic_{idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {output_path}")
    plt.close()


def main():
    """Main diagnostic function"""
    
    # Settings
    model_path = 'model/upview model/segmentation/segmentation.pt'
    data_dir = Path('data/upview data')
    
    # Get first few images
    image_files = sorted(list(data_dir.glob('*.jpg')))[:3]
    
    if not image_files:
        print("❌ No images found!")
        return
    
    print("="*60)
    print("🔬 SEGMENTATION MASK DIAGNOSTIC")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"Analyzing first 3 images from: {data_dir}")
    
    # Analyze each image
    for img_path in image_files:
        diagnose_segmentation(str(img_path), model_path)
    
    print("\n" + "="*60)
    print("📊 ANALYSIS COMPLETE")
    print("="*60)
    print("\nCheck the saved visualizations (mask_diagnostic_*.png)")
    print("\n💡 INTERPRETATION:")
    print("   • If polygon has 3 vertices → Triangle shape detected ✅")
    print("   • If polygon has 4+ vertices → Shape is not triangular ⚠️")
    print("   • Check mask confidence heatmap for quality")
    print("   • Low confidence areas may need more training data")
    print("="*60)


if __name__ == '__main__':
    main()

