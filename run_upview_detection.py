#!/usr/bin/env python3
"""
Upview Detection Model - Image Processing
Process upview images with the detection model
Shows bounding boxes and measurements
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from secure_loader import load_protected_model


def load_calibration(filename='calibration_upview.json'):
    """Load calibration data for upview camera"""
    if not os.path.exists(filename):
        # Try old calibration file as fallback
        if os.path.exists('calibration.json'):
            print("⚠️  Using old calibration.json - consider calibrating for upview camera")
            filename = 'calibration.json'
        else:
            return None
    try:
        with open(filename, 'r') as f:
            cal_data = json.load(f)
            if 'camera' in cal_data and cal_data['camera'] != 'upview':
                print(f"⚠️  Warning: Using {cal_data.get('camera', 'unknown')} camera calibration for upview")
            return cal_data
    except:
        return None


def run_upview_detection(model_path, data_dir, output_dir='output_upview_detection'):
    """
    Run detection on upview data
    
    Args:
        model_path: Path to detection model
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
    
    # Load model (supports protected models)
    print(f"\n🤖 Loading detection model: {model_path}")
    if 'protected_' in model_path or os.path.exists(model_path + '.lock'):
        model = load_protected_model(model_path)
    else:
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
        
        # Run detection
        results = model(str(img_path), verbose=False)
        
        detection_count = 0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"  ✅ Detections: {len(result.boxes)} objects")
                
                for idx, box in enumerate(result.boxes):
                    detection_count += 1
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    # Calculate dimensions
                    width_px = x2 - x1
                    height_px = y2 - y1
                    
                    # Color
                    color = (50, 255, 150)
                    
                    # Draw filled rectangle
                    cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color, -1)
                    
                    # Draw bounding box
                    cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 3)
                    
                    # Add measurements if available
                    if mm_per_pixel:
                        height_mm = height_px * mm_per_pixel
                        
                        # Draw vertical measurement line
                        mid_x = (x1 + x2) // 2
                        cv2.line(img_result, (mid_x, y1), (mid_x, y2), (0, 255, 255), 2)
                        cv2.circle(img_result, (mid_x, y1), 5, (0, 255, 255), -1)
                        cv2.circle(img_result, (mid_x, y2), 5, (0, 255, 255), -1)
                        
                        # Measurement text
                        height_text = f"{height_mm:.1f}mm"
                        (tw, th), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        text_x = mid_x + 15
                        text_y = (y1 + y2) // 2
                        
                        # Background for text
                        cv2.rectangle(img_result, (text_x - 5, text_y - th - 5),
                                     (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
                        cv2.putText(img_result, height_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        print(f"    - {cls_name}: {conf:.2f} | Height: {height_mm:.1f}mm ({height_px}px)")
                    else:
                        print(f"    - {cls_name}: {conf:.2f} | Height: {height_px}px")
                    
                    # Add class label
                    label = f"{cls_name} {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img_result, (x1, y1 - lh - 15), (x1 + lw + 10, y1), color, -1)
                    cv2.putText(img_result, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                print(f"  ⚠️  No objects detected")
        
        # Blend overlay
        img_result = cv2.addWeighted(img_result, 0.7, img_overlay, 0.3, 0)
        
        # Save result
        output_path = os.path.join(output_dir, f'det_{img_path.name}')
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
    model_path = 'model/upview model/detection/detection.pt'
    data_dir = 'data/upview data'
    output_dir = 'output_upview_detection'
    
    # Check model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Check data
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print("=" * 60)
    print("🎯 UPVIEW DETECTION MODEL")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    # Run detection
    run_upview_detection(model_path, data_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("✨ Detection complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

