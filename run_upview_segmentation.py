#!/usr/bin/env python3
"""
Upview Segmentation Model - Image Processing
Process upview images with the segmentation model
Shows pixel-perfect masks and measurements
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


def load_calibration(filename='calibration.json'):
    """Load calibration data"""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None


def run_upview_segmentation(model_path, data_dir, output_dir='output_upview_segmentation'):
    """
    Run segmentation on upview data
    
    Args:
        model_path: Path to segmentation model
        data_dir: Directory with upview images
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load calibration
    calibration = load_calibration()
    if calibration:
        mm_per_pixel = calibration['mm_per_pixel']
        print(f"✅ Calibration loaded: {mm_per_pixel:.6f} mm/pixel")
        print(f"   Based on: {calibration['mm_distance']:.2f}mm = {calibration['pixel_distance']:.2f}px")
    else:
        mm_per_pixel = None
        print("⚠️  No calibration found. Using pixel measurements.")
    
    # Load model
    print(f"\n🤖 Loading segmentation model: {model_path}")
    model = YOLO(model_path)
    print(f"   Model task: {model.task}")
    
    # Get images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(data_dir).glob(f'*{ext}'))
        image_files.extend(Path(data_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    print(f"\n📁 Found {len(image_files)} images in {data_dir}")
    
    if not image_files:
        print("❌ No images found!")
        return
    
    # Process each image
    results_list = []
    print(f"\n🔄 Processing images...")
    print("-" * 60)
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        img_result = img.copy()
        img_overlay = img.copy()
        
        # Run segmentation
        results = model(str(img_path), verbose=False)
        
        detection_count = 0
        
        for result in results:
            # Check for segmentation masks
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                print(f"  ✅ Segmentations: {len(result.masks)} objects")
                
                for idx, (mask, box) in enumerate(zip(result.masks, result.boxes)):
                    detection_count += 1
                    
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    height_px = y2 - y1
                    width_px = x2 - x1
                    
                    # Get mask
                    mask_array = mask.data[0].cpu().numpy()
                    mask_resized = cv2.resize(mask_array, (img.shape[1], img.shape[0]))
                    mask_bool = mask_resized > 0.5
                    
                    # Calculate mask area
                    mask_area = mask_bool.sum()
                    
                    # Generate color (unique per instance)
                    np.random.seed(idx + 42)  # Consistent colors
                    color = tuple(map(int, np.random.randint(50, 255, 3)))
                    
                    # Apply colored mask
                    img_overlay[mask_bool] = color
                    
                    # Draw contour
                    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), 
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_result, contours, -1, color, 3)
                    
                    # Add measurements if available
                    if mm_per_pixel:
                        height_mm = height_px * mm_per_pixel
                        
                        # Draw vertical measurement line
                        mid_x = (x1 + x2) // 2
                        cv2.line(img_result, (mid_x, y1), (mid_x, y2), (0, 255, 255), 2)
                        cv2.circle(img_result, (mid_x, y1), 5, (0, 255, 255), -1)
                        cv2.circle(img_result, (mid_x, y2), 5, (0, 255, 255), -1)
                        
                        # Measurement text
                        height_text = f"{height_mm:.2f}mm"
                        (tw, th), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        text_x = mid_x + 15
                        text_y = (y1 + y2) // 2
                        
                        # Background for text
                        cv2.rectangle(img_result, (text_x - 5, text_y - th - 5),
                                     (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
                        cv2.putText(img_result, height_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        print(f"    - {cls_name}: {conf:.2f} | Height: {height_mm:.2f}mm ({height_px}px) | Mask: {mask_area} pixels")
                    else:
                        print(f"    - {cls_name}: {conf:.2f} | Height: {height_px}px | Mask: {mask_area} pixels")
                    
                    # Add class label
                    label = f"{cls_name} {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img_result, (x1, y1 - lh - 15), (x1 + lw + 10, y1), color, -1)
                    cv2.putText(img_result, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            elif result.boxes is not None and len(result.boxes) > 0:
                print(f"  ⚠️  No masks found, using bounding boxes")
                for box in result.boxes:
                    detection_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    color = (255, 100, 50)
                    cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(img_result, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    print(f"    - {cls_name}: {conf:.2f}")
        
        if detection_count == 0:
            print(f"  ⚠️  No objects detected")
        
        # Blend overlay
        img_result = cv2.addWeighted(img_result, 0.6, img_overlay, 0.4, 0)
        
        # Save result
        output_path = os.path.join(output_dir, f'seg_{img_path.name}')
        cv2.imwrite(output_path, img_result)
        print(f"  💾 Saved: {output_path}")
        
        # Store for summary
        results_list.append((img_path.name, img_result))
    
    # Create summary grid
    print(f"\n📊 Creating summary visualization...")
    display_results(results_list, output_dir)
    
    print(f"\n✅ All results saved to: {output_dir}/")
    print(f"✅ Summary visualization: {output_dir}/summary.png")


def display_results(results_list, output_dir):
    """Create summary grid"""
    num_images = len(results_list)
    if num_images == 0:
        return
    
    # Calculate grid size
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if num_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Plot each result
    for idx, (img_name, img) in enumerate(results_list):
        if idx < len(axes):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(img_name, fontsize=10)
            axes[idx].axis('off')
    
    # Hide unused subplots
    if isinstance(axes, (list, np.ndarray)):
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    summary_path = os.path.join(output_dir, 'summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"  💾 Summary saved: {summary_path}")
    plt.close()


def main():
    """Main function"""
    model_path = 'model/upview model/segmentation/segmentation.pt'
    data_dir = 'data/upview data'
    output_dir = 'output_upview_segmentation'
    
    # Check model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Check data
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print("=" * 60)
    print("🎨 UPVIEW SEGMENTATION MODEL")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    # Run segmentation
    run_upview_segmentation(model_path, data_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("✨ Segmentation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

