#!/usr/bin/env python3
"""
Test NEW Retrained Segmentation Model
Compare old vs new segmentation model
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


def process_with_model(image_path, model, mm_per_pixel=None):
    """Process single image and return detailed info"""
    img = cv2.imread(str(image_path))
    img_result = img.copy()
    img_overlay = img.copy()
    
    # Run segmentation
    results = model(str(image_path), verbose=False)
    
    detections = []
    mask_vertices = []
    
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
            for idx, (mask, box) in enumerate(zip(result.masks, result.boxes)):
                # Get class info
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                height_px = y2 - y1
                
                # Get mask
                mask_array = mask.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_array, (img.shape[1], img.shape[0]))
                mask_bool = mask_resized > 0.5
                mask_area = mask_bool.sum()
                
                # Find contours
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Get polygon approximation
                vertices_count = 0
                if contours:
                    main_contour = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(main_contour, True)
                    epsilon = 0.02 * perimeter
                    approx = cv2.approxPolyDP(main_contour, epsilon, True)
                    vertices_count = len(approx)
                
                # Color
                np.random.seed(idx + 42)
                color = tuple(map(int, np.random.randint(50, 255, 3)))
                
                # Apply mask
                img_overlay[mask_bool] = color
                
                # Draw contour
                if contours:
                    cv2.drawContours(img_result, contours, -1, color, 3)
                
                # Add measurements
                if mm_per_pixel:
                    height_mm = height_px * mm_per_pixel
                    
                    # Draw measurement line
                    mid_x = (x1 + x2) // 2
                    cv2.line(img_result, (mid_x, y1), (mid_x, y2), (0, 255, 255), 2)
                    cv2.circle(img_result, (mid_x, y1), 5, (0, 255, 255), -1)
                    cv2.circle(img_result, (mid_x, y2), 5, (0, 255, 255), -1)
                    
                    # Measurement text
                    height_text = f"{height_mm:.1f}mm"
                    (tw, th), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_x = mid_x + 15
                    text_y = (y1 + y2) // 2
                    cv2.rectangle(img_result, (text_x - 5, text_y - th - 5),
                                 (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
                    cv2.putText(img_result, height_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Add class label
                label = f"{cls_name} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img_result, (x1, y1 - lh - 15), (x1 + lw + 10, y1), color, -1)
                cv2.putText(img_result, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'height_px': height_px,
                    'mask_area': mask_area,
                    'vertices': vertices_count
                })
    
    # Blend overlay
    img_result = cv2.addWeighted(img_result, 0.6, img_overlay, 0.4, 0)
    
    return img_result, detections


def main():
    """Test new segmentation model"""
    
    print("=" * 70)
    print("🆕 TESTING NEW RETRAINED SEGMENTATION MODEL")
    print("=" * 70)
    
    # Paths
    old_model_path = 'model/upview model/segmentation/segmentation.pt'
    new_model_path = 'model/upview model/segmentation/segmentationnew.pt'
    data_dir = Path('data/upview data')
    output_dir = 'output_new_segmentation'
    
    # Create output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/comparison', exist_ok=True)
    
    # Load calibration
    calibration = load_calibration()
    mm_per_pixel = calibration['mm_per_pixel'] if calibration else None
    
    if mm_per_pixel:
        print(f"\n✅ Calibration loaded: {mm_per_pixel:.6f} mm/pixel")
    
    # Load models
    print(f"\n📦 Loading models...")
    print(f"   Old model: {old_model_path}")
    old_model = YOLO(old_model_path)
    print(f"   ✅ Old model loaded")
    
    print(f"   New model: {new_model_path}")
    new_model = YOLO(new_model_path)
    print(f"   ✅ New model loaded")
    
    # Get images
    image_files = sorted(list(data_dir.glob('*.jpg')))
    print(f"\n📁 Found {len(image_files)} images")
    
    # Statistics
    old_stats = {'vertices': [], 'confidence': [], 'mask_area': []}
    new_stats = {'vertices': [], 'confidence': [], 'mask_area': []}
    
    print(f"\n🔄 Processing and comparing...")
    print("-" * 70)
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_path.name}")
        
        # Process with old model
        old_result, old_detections = process_with_model(img_path, old_model, mm_per_pixel)
        
        # Process with new model
        new_result, new_detections = process_with_model(img_path, new_model, mm_per_pixel)
        
        # Load original
        original = cv2.imread(str(img_path))
        
        # Create comparison
        h, w = original.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, 0:w] = original
        comparison[:, w:w*2] = old_result
        comparison[:, w*2:w*3] = new_result
        
        # Add labels
        labels = ['Original', 'Old Model', 'NEW Model ✨']
        positions = [w//2, w + w//2, 2*w + w//2]
        colors = [(255, 255, 255), (255, 200, 100), (100, 255, 100)]
        
        for label, pos, color in zip(labels, positions, colors):
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            x = pos - tw // 2
            cv2.rectangle(comparison, (x - 10, 10), (x + tw + 10, 55), (0, 0, 0), -1)
            cv2.putText(comparison, label, (x, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Save comparison
        comp_path = f'{output_dir}/comparison/comp_{img_path.name}'
        cv2.imwrite(comp_path, comparison)
        
        # Save new model result
        new_path = f'{output_dir}/new_{img_path.name}'
        cv2.imwrite(new_path, new_result)
        
        # Print comparison
        print(f"  📊 Old Model:")
        for det in old_detections:
            print(f"     {det['class']}: conf={det['confidence']:.3f}, vertices={det['vertices']}, area={det['mask_area']}")
            old_stats['vertices'].append(det['vertices'])
            old_stats['confidence'].append(det['confidence'])
            old_stats['mask_area'].append(det['mask_area'])
        
        print(f"  📊 NEW Model:")
        for det in new_detections:
            print(f"     {det['class']}: conf={det['confidence']:.3f}, vertices={det['vertices']}, area={det['mask_area']}")
            new_stats['vertices'].append(det['vertices'])
            new_stats['confidence'].append(det['confidence'])
            new_stats['mask_area'].append(det['mask_area'])
        
        # Improvement indicator
        if new_detections and old_detections:
            if new_detections[0]['vertices'] < old_detections[0]['vertices']:
                print(f"  ✅ IMPROVEMENT: {new_detections[0]['vertices']} vs {old_detections[0]['vertices']} vertices (closer to triangle!)")
            elif new_detections[0]['vertices'] == 3:
                print(f"  🎯 PERFECT: Triangle shape achieved!")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("📊 OVERALL COMPARISON")
    print("=" * 70)
    
    print(f"\n🔴 OLD MODEL:")
    print(f"   Average vertices: {np.mean(old_stats['vertices']):.1f}")
    print(f"   Average confidence: {np.mean(old_stats['confidence']):.3f}")
    print(f"   Vertex range: {min(old_stats['vertices'])}-{max(old_stats['vertices'])}")
    
    print(f"\n🟢 NEW MODEL:")
    print(f"   Average vertices: {np.mean(new_stats['vertices']):.1f}")
    print(f"   Average confidence: {np.mean(new_stats['confidence']):.3f}")
    print(f"   Vertex range: {min(new_stats['vertices'])}-{max(new_stats['vertices'])}")
    
    # Verdict
    print(f"\n" + "=" * 70)
    print("💡 VERDICT")
    print("=" * 70)
    
    old_avg_vert = np.mean(old_stats['vertices'])
    new_avg_vert = np.mean(new_stats['vertices'])
    
    if new_avg_vert <= 3.5 and old_avg_vert > 3.5:
        print("🎉 EXCELLENT! New model produces much better triangle shapes!")
    elif new_avg_vert < old_avg_vert:
        print(f"✅ IMPROVED! Vertices reduced from {old_avg_vert:.1f} to {new_avg_vert:.1f}")
        print(f"   {((old_avg_vert - new_avg_vert) / old_avg_vert * 100):.1f}% improvement")
    elif new_avg_vert == old_avg_vert:
        print("😐 SIMILAR: No significant change in mask shape")
    else:
        print("⚠️  Mask quality may need more work")
    
    print(f"\n📁 Results saved to:")
    print(f"   • Individual: {output_dir}/new_*.jpg")
    print(f"   • Comparisons: {output_dir}/comparison/comp_*.jpg")
    print("=" * 70)


if __name__ == '__main__':
    main()

