#!/usr/bin/env python3
"""
Test script to verify setup
"""

import os
import sys

print("=" * 60)
print("🧪 Testing YOLO Measurement Setup")
print("=" * 60)

# Check model
if os.path.exists('model/my_model.pt'):
    print("✅ Model found: model/my_model.pt")
else:
    print("❌ Model NOT found: model/my_model.pt")

# Check data directory
data_files = []
if os.path.exists('data'):
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        import glob
        data_files.extend(glob.glob(f'data/*{ext}'))
        data_files.extend(glob.glob(f'data/*{ext.upper()}'))
    print(f"✅ Data folder found with {len(data_files)} images")
else:
    print("❌ Data folder NOT found")

# Check scripts
scripts = ['calibrate.py', 'run_segmentation.py', 'run_detection.py']
for script in scripts:
    if os.path.exists(script):
        print(f"✅ Script found: {script}")
    else:
        print(f"❌ Script NOT found: {script}")

# Check calibration
if os.path.exists('calibration.json'):
    print("✅ Calibration file found: calibration.json")
    import json
    with open('calibration.json', 'r') as f:
        cal = json.load(f)
    print(f"   📏 {cal['mm_distance']:.2f}mm = {cal['pixel_distance']:.2f}px")
    print(f"   🎯 Ratio: {cal['mm_per_pixel']:.4f} mm/pixel")
else:
    print("⚠️  Calibration file NOT found")
    print("   Run 'python calibrate.py' to create it")

# Check packages
print("\n📦 Checking packages...")
try:
    import cv2
    print(f"✅ opencv-python: {cv2.__version__}")
except:
    print("❌ opencv-python not installed")

try:
    from ultralytics import YOLO
    print("✅ ultralytics: installed")
except:
    print("❌ ultralytics not installed")

try:
    import torch
    print(f"✅ torch: {torch.__version__}")
except:
    print("❌ torch not installed")

print("\n" + "=" * 60)
print("🎯 Next Steps:")
print("=" * 60)

if not os.path.exists('calibration.json'):
    print("1. Run calibration:")
    print("   python calibrate.py")
    print("\n2. Then run detection with measurements:")
    print("   python run_segmentation.py")
else:
    print("✅ You're ready to go!")
    print("   Run: python run_segmentation.py")

print("=" * 60)

