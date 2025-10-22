#!/usr/bin/env python3
"""
YOLO Segmentation Script with Measurements
Uses YOLOv8 segmentation model or converts detection to segmentation visualization
Displays measurements in mm using calibration data
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from secure_loader import load_protected_model

def load_calibration(filename='calibration.json'):
    """Load calibration data"""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None

def run_yolo_segmentation(model_path, data_dir, output_dir='output_segmentation', use_pretrained=False):
    """
    Run YOLO segmentation inference on all images in the data directory
    
    Args:
        model_path: Path to the YOLO model (.pt file)
        data_dir: Directory containing test images
        output_dir: Directory to save output images
        use_pretrained: If True, use YOLOv8 segmentation model as fallback
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load calibration
    calibration = load_calibration()
    if calibration:
        mm_per_pixel = calibration['mm_per_pixel']
        print(f"✅ Calibration loaded: {mm_per_pixel:.4f} mm/pixel")
        print(f"   Based on: {calibration['mm_distance']:.2f}mm = {calibration['pixel_distance']:.2f}px")
    else:
        mm_per_pixel = None
        print("⚠️  No calibration found. Run 'python calibrate.py' first to enable measurements.")
        print("   Continuing without measurements...")
    
    # Load the YOLO model (handles both protected and regular models)
    print(f"Loading model from: {model_path}")
    
    # Check if it's a protected model
    if 'protected_' in model_path or os.path.exists(model_path + '.lock'):
        model = load_protected_model(model_path)
    else:
        model = YOLO(model_path)
    
    # Check if model supports segmentation
    print(f"Model task: {model.task}")
    
    if model.task != 'segment':
        if use_pretrained:
            print("⚠️  Your model is a detection model, not segmentation.")
            print("🔄 Loading YOLOv8n-seg (segmentation) model instead...")
            model = YOLO('yolov8n-seg.pt')  # This will auto-download
        else:
            print("⚠️  Your model is a detection model, not a segmentation model.")
            print("💡 To use segmentation, you need a model trained with segmentation labels.")
            print("   We'll create enhanced visualizations with filled boxes instead.")
    
    # Get all image files from data directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(data_dir).glob(f'*{ext}'))
        image_files.extend(Path(data_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in {data_dir}")
    
    if not image_files:
        print("No images found! Please check the data directory.")
        return
    
    # Process each image
    results_list = []
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        # Run inference
        results = model(str(img_path))
        
        # Save the annotated image
        for i, result in enumerate(results):
            # Create custom visualization
            img = cv2.imread(str(img_path))
            img_overlay = img.copy()
            
            # Check for segmentation masks
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                print(f"  Segmentations: {len(result.masks)} objects")
                
                # Draw each mask
                for j, (mask, box) in enumerate(zip(result.masks, result.boxes)):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                    
                    # Get mask data
                    mask_array = mask.data[0].cpu().numpy()
                    mask_resized = cv2.resize(mask_array, (img.shape[1], img.shape[0]))
                    mask_bool = mask_resized > 0.5
                    
                    # Generate random color for this instance
                    color = np.random.randint(0, 255, 3).tolist()
                    
                    # Apply colored mask
                    img_overlay[mask_bool] = img_overlay[mask_bool] * 0.5 + np.array(color) * 0.5
                    
                    # Calculate mask area
                    mask_area = mask_bool.sum()
                    print(f"    - {cls_name}: {conf:.2f} (mask area: {mask_area:.0f} pixels)")
                
                # Blend the overlay
                img = cv2.addWeighted(img, 0.5, img_overlay, 0.5, 0)
                
            elif result.boxes is not None and len(result.boxes) > 0:
                print(f"  Detections: {len(result.boxes)} objects (no masks available)")
                print(f"  Creating pseudo-segmentation with filled regions...")
                
                # For detection models, create filled box regions as pseudo-segmentation
                for idx, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Calculate dimensions
                    width_px = x2 - x1
                    height_px = y2 - y1
                    
                    # Generate random color
                    color = np.random.randint(50, 255, 3).tolist()
                    
                    # Create semi-transparent filled rectangle
                    cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color, -1)
                    
                    # Draw contour
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Calculate measurements if calibration available
                    if mm_per_pixel:
                        height_mm = height_px * mm_per_pixel
                        width_mm = width_px * mm_per_pixel
                        
                        # Draw vertical measurement line
                        mid_x = (x1 + x2) // 2
                        cv2.line(img, (mid_x, y1), (mid_x, y2), (0, 255, 255), 2)
                        cv2.circle(img, (mid_x, y1), 4, (0, 255, 255), -1)
                        cv2.circle(img, (mid_x, y2), 4, (0, 255, 255), -1)
                        
                        # Add measurement text - vertical height
                        height_text = f"{height_mm:.1f}mm"
                        (tw, th), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Background for measurement text
                        text_x = mid_x + 10
                        text_y = (y1 + y2) // 2
                        cv2.rectangle(img, (text_x - 5, text_y - th - 5), 
                                     (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
                        cv2.putText(img, height_text, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Add label with class name and confidence
                        label = f"{cls_name} {conf:.2f}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
                        cv2.putText(img, label, (x1 + 5, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        print(f"    - {cls_name}: {conf:.2f} | Height: {height_mm:.1f}mm ({height_px}px)")
                    else:
                        # No calibration - just show label
                        label = f"{cls_name} {conf:.2f}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        cv2.putText(img, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        print(f"    - {cls_name}: {conf:.2f} | Height: {height_px}px")
                
                # Blend the overlay
                img = cv2.addWeighted(img, 0.7, img_overlay, 0.3, 0)
            else:
                print(f"  No objects detected")
            
            # Save result
            output_path = os.path.join(output_dir, f'seg_{img_path.name}')
            cv2.imwrite(output_path, img)
            print(f"  Saved segmentation to: {output_path}")
            
            # Store for visualization
            results_list.append((img_path.name, img))
    
    # Display results in a grid
    display_results(results_list, output_dir)
    
    print(f"\n✅ All results saved to: {output_dir}/")
    print(f"✅ Summary visualization saved to: {output_dir}/summary.png")

def display_results(results_list, output_dir):
    """
    Display all results in a matplotlib grid
    
    Args:
        results_list: List of (filename, image) tuples
        output_dir: Directory to save the summary image
    """
    num_images = len(results_list)
    if num_images == 0:
        return
    
    # Calculate grid size
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    # Plot each result
    for idx, (img_name, img) in enumerate(results_list):
        if idx < len(axes):
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(img_name, fontsize=10)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the summary
    summary_path = os.path.join(output_dir, 'summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Summary visualization saved to: {summary_path}")
    
    # Show the plot
    plt.show()

def main():
    """Main function"""
    # Paths
    model_path = 'model/protected_my_model.pt'
    data_dir = 'data'
    output_dir = 'output_segmentation'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found at {data_dir}")
        return
    
    print("=" * 60)
    print("🎨 YOLO Instance Segmentation")
    print("=" * 60)
    print("\nOptions:")
    print("1. Use your fine-tuned model (will create pseudo-segmentation)")
    print("2. Use YOLOv8n-seg pretrained model (true segmentation)")
    print("\nRunning with your fine-tuned model...")
    print("-" * 60)
    
    # Run inference with your model
    run_yolo_segmentation(model_path, data_dir, output_dir, use_pretrained=False)
    
    print("\n" + "=" * 60)
    print("✨ Segmentation complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()

