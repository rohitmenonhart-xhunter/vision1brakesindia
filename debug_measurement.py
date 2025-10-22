#!/usr/bin/env python3
"""
Debug script to investigate measurement accuracy
"""

import cv2
import json
from ultralytics import YOLO

# Load calibration
with open('calibration.json', 'r') as f:
    cal = json.load(f)

print("=" * 60)
print("🔍 MEASUREMENT DEBUG")
print("=" * 60)

print("\n📏 Calibration Data:")
print(f"  Pixel distance: {cal['pixel_distance']:.2f} px")
print(f"  MM distance: {cal['mm_distance']:.2f} mm")
print(f"  MM per pixel: {cal['mm_per_pixel']:.6f} mm/px")
print(f"  Image used: {cal['calibration_image']}")

# Load the calibration image
cal_img = cv2.imread(cal['calibration_image'])
print(f"\n🖼️  Calibration image size: {cal_img.shape[1]}x{cal_img.shape[0]} (WxH)")

# Load model and run inference
print("\n🤖 Running YOLO on calibration image...")
model = YOLO('model/my_model.pt')
results = model(cal['calibration_image'])

print(f"\n📊 YOLO Results:")
for result in results:
    print(f"  Original image size: {result.orig_shape}")
    print(f"  Inference image size: {result.boxes.orig_shape if hasattr(result.boxes, 'orig_shape') else 'N/A'}")
    
    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            height_px = y2 - y1
            width_px = x2 - x1
            
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            
            print(f"\n  Detection {i+1}:")
            print(f"    Class: {cls_name}")
            print(f"    Confidence: {conf:.2f}")
            print(f"    Bounding box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
            print(f"    Width (pixels): {width_px:.2f} px")
            print(f"    Height (pixels): {height_px:.2f} px")
            
            # Calculate with current calibration
            height_mm = height_px * cal['mm_per_pixel']
            width_mm = width_px * cal['mm_per_pixel']
            
            print(f"    Height (mm): {height_mm:.2f} mm")
            print(f"    Width (mm): {width_mm:.2f} mm")
            
            # If this is showing 8.7 instead of 8.25
            if abs(height_mm - 8.7) < 0.5:
                print(f"\n  ⚠️  FOUND THE ISSUE!")
                print(f"    Current calculation: {height_px:.2f} px × {cal['mm_per_pixel']:.6f} = {height_mm:.2f} mm")
                print(f"    Expected: 8.25 mm")
                print(f"    Actual: {height_mm:.2f} mm")
                print(f"    Error: {height_mm - 8.25:.2f} mm ({((height_mm - 8.25) / 8.25 * 100):.1f}%)")
                
                # Calculate what the calibration should be
                correct_mm_per_pixel = 8.25 / height_px
                print(f"\n  💡 CORRECTION:")
                print(f"    To get 8.25mm from {height_px:.2f}px:")
                print(f"    mm_per_pixel should be: {correct_mm_per_pixel:.6f}")
                print(f"    Current mm_per_pixel: {cal['mm_per_pixel']:.6f}")
                print(f"    Difference: {(correct_mm_per_pixel - cal['mm_per_pixel']):.6f}")

print("\n" + "=" * 60)
print("🔍 Analysis Complete")
print("=" * 60)

