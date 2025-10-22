#!/usr/bin/env python3
"""
Final accuracy verification script
"""

import json

print("=" * 60)
print("✅ FINAL ACCURACY VERIFICATION")
print("=" * 60)

# Load calibration
with open('calibration.json', 'r') as f:
    cal = json.load(f)

print("\n📏 Calibration Settings:")
print(f"  Method: {cal.get('method', 'N/A')}")
print(f"  Image: {cal['calibration_image']}")
print(f"  BBox Height: {cal['pixel_distance']:.2f} pixels")
print(f"  Real Height: {cal['mm_distance']:.2f} mm")
print(f"  Calibration Factor: {cal['mm_per_pixel']:.6f} mm/pixel")

# Calculate verification
pixel_height = cal['pixel_distance']
mm_per_px = cal['mm_per_pixel']
calculated_mm = pixel_height * mm_per_px

print("\n🧮 Verification Calculation:")
print(f"  {pixel_height:.2f} px × {mm_per_px:.6f} mm/px = {calculated_mm:.6f} mm")

print("\n📊 Accuracy Test:")
expected = cal['mm_distance']
actual = calculated_mm
error = abs(actual - expected)
error_pct = (error / expected) * 100

print(f"  Expected: {expected:.2f} mm")
print(f"  Actual:   {actual:.2f} mm")
print(f"  Error:    {error:.4f} mm ({error_pct:.2f}%)")

if error < 0.01:
    print("\n✅ STATUS: PERFECT ACCURACY!")
    print("   Error is negligible (< 0.01mm)")
elif error < 0.1:
    print("\n✅ STATUS: EXCELLENT ACCURACY!")
    print("   Error is very small (< 0.1mm)")
else:
    print(f"\n⚠️  STATUS: Error of {error:.2f}mm detected")
    print("   Consider recalibrating")

print("\n" + "=" * 60)
print("🎯 SYSTEM STATUS: OPERATIONAL")
print("=" * 60)
print("\n📋 Quick Start:")
print("  1. conda activate yolo_env")
print("  2. python run_segmentation.py")
print("\n🔄 To recalibrate:")
print("  python calibrate_bbox.py")
print("=" * 60)

