#!/usr/bin/env python3
"""
YOLO Object Detection and Segmentation Script
Runs inference on test images and displays results
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from secure_loader import load_protected_model

def run_yolo_inference(model_path, data_dir, output_dir='output'):
    """
    Run YOLO segmentation inference on all images in the data directory
    
    Args:
        model_path: Path to the YOLO model (.pt file)
        data_dir: Directory containing test images
        output_dir: Directory to save output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the YOLO model (supports protected models)
    print(f"Loading model from: {model_path}")
    if 'protected_' in model_path or os.path.exists(model_path + '.lock'):
        model = load_protected_model(model_path)
    else:
        model = YOLO(model_path)
    
    # Check if model supports segmentation
    print(f"Model task: {model.task}")
    if model.task != 'segment':
        print("⚠️  Warning: Model may not support segmentation. Will attempt to run anyway.")
    
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
        
        # Run inference with segmentation
        results = model(str(img_path), task='segment')
        
        # Save the annotated image
        for i, result in enumerate(results):
            # Plot and save with segmentation masks
            output_path = os.path.join(output_dir, f'result_{img_path.name}')
            result.save(filename=output_path)
            print(f"  Saved result to: {output_path}")
            
            # Print segmentation info
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                print(f"  Segmentations: {len(result.masks)} objects")
                for j, (mask, box) in enumerate(zip(result.masks, result.boxes)):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                    # Calculate mask area
                    mask_area = mask.data.sum().item() if hasattr(mask, 'data') else 0
                    print(f"    - {cls_name}: {conf:.2f} (mask area: {mask_area:.0f} pixels)")
            elif result.boxes is not None and len(result.boxes) > 0:
                print(f"  Detections (boxes only): {len(result.boxes)} objects")
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                    print(f"    - {cls_name}: {conf:.2f}")
            else:
                print(f"  No objects detected")
            
            # Store for visualization
            results_list.append((img_path.name, result))
    
    # Display results in a grid
    display_results(results_list, output_dir)
    
    print(f"\n✅ All results saved to: {output_dir}/")
    print(f"✅ Summary visualization saved to: {output_dir}/summary.png")

def display_results(results_list, output_dir):
    """
    Display all results in a matplotlib grid
    
    Args:
        results_list: List of (filename, result) tuples
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
    for idx, (img_name, result) in enumerate(results_list):
        if idx < len(axes):
            # Get the annotated image
            img = result.plot()
            
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
    model_path = 'model/my_model.pt'
    data_dir = 'data'
    output_dir = 'output'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found at {data_dir}")
        return
    
    print("=" * 60)
    print("🚀 YOLO Instance Segmentation")
    print("=" * 60)
    
    # Run inference
    run_yolo_inference(model_path, data_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("✨ Inference complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()

