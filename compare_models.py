#!/usr/bin/env python3
"""
Model Comparison: Detection vs Segmentation
Runs both models on upview data and creates side-by-side comparison
"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
import time


def load_calibration(filename='calibration.json'):
    """Load calibration data"""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None


def process_with_detection_model(image_path, model, mm_per_pixel=None):
    """Process image with detection model"""
    img = cv2.imread(image_path)
    img_result = img.copy()
    img_overlay = img.copy()
    
    # Run detection
    start_time = time.time()
    results = model(image_path, verbose=False)
    inference_time = (time.time() - start_time) * 1000
    
    detections = []
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                
                height_px = y2 - y1
                width_px = x2 - x1
                
                # Color
                color = (50, 255, 150)
                
                # Draw filled rectangle
                cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color, -1)
                
                # Draw bounding box
                cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 3)
                
                # Add measurements if available
                if mm_per_pixel:
                    height_mm = height_px * mm_per_pixel
                    
                    # Draw measurement line
                    mid_x = (x1 + x2) // 2
                    cv2.line(img_result, (mid_x, y1), (mid_x, y2), (0, 255, 255), 2)
                    cv2.circle(img_result, (mid_x, y1), 4, (0, 255, 255), -1)
                    cv2.circle(img_result, (mid_x, y2), 4, (0, 255, 255), -1)
                    
                    # Measurement text
                    height_text = f"{height_mm:.1f}mm"
                    text_x = mid_x + 10
                    text_y = (y1 + y2) // 2
                    cv2.putText(img_result, height_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    measurement = f"{height_mm:.1f}mm"
                else:
                    measurement = f"{height_px}px"
                
                # Class label
                label = f"{cls_name} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_result, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
                cv2.putText(img_result, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'measurement': measurement,
                    'bbox': (x1, y1, x2, y2)
                })
    
    # Blend overlay
    img_result = cv2.addWeighted(img_result, 0.7, img_overlay, 0.3, 0)
    
    return img_result, detections, inference_time


def process_with_segmentation_model(image_path, model, mm_per_pixel=None):
    """Process image with segmentation model"""
    img = cv2.imread(image_path)
    img_result = img.copy()
    img_overlay = img.copy()
    
    # Run segmentation
    start_time = time.time()
    results = model(image_path, verbose=False)
    inference_time = (time.time() - start_time) * 1000
    
    detections = []
    
    for result in results:
        # Check for segmentation masks
        if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
            for mask, box in zip(result.masks, result.boxes):
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
                
                # Color
                color = (255, 100, 50)
                
                # Apply mask
                img_overlay[mask_bool] = color
                
                # Draw contour
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8), 
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_result, contours, -1, color, 3)
                
                # Add measurements if available
                if mm_per_pixel:
                    height_mm = height_px * mm_per_pixel
                    
                    # Draw measurement line
                    mid_x = (x1 + x2) // 2
                    cv2.line(img_result, (mid_x, y1), (mid_x, y2), (0, 255, 255), 2)
                    cv2.circle(img_result, (mid_x, y1), 4, (0, 255, 255), -1)
                    cv2.circle(img_result, (mid_x, y2), 4, (0, 255, 255), -1)
                    
                    # Measurement text
                    height_text = f"{height_mm:.1f}mm"
                    text_x = mid_x + 10
                    text_y = (y1 + y2) // 2
                    cv2.putText(img_result, height_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    measurement = f"{height_mm:.1f}mm"
                else:
                    measurement = f"{height_px}px"
                
                # Class label
                label = f"{cls_name} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_result, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
                cv2.putText(img_result, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'measurement': measurement,
                    'mask_area': mask_bool.sum()
                })
        
        # If no masks, fall back to boxes
        elif result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                
                height_px = y2 - y1
                color = (255, 100, 50)
                
                cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 3)
                
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(img_result, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'measurement': f"{height_px}px"
                })
    
    # Blend overlay
    img_result = cv2.addWeighted(img_result, 0.6, img_overlay, 0.4, 0)
    
    return img_result, detections, inference_time


def create_comparison_image(original, detection_result, segmentation_result, 
                            detection_info, segmentation_info, image_name):
    """Create side-by-side comparison"""
    h, w = original.shape[:2]
    
    # Create comparison canvas
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # Place images
    comparison[:, 0:w] = original
    comparison[:, w:w*2] = detection_result
    comparison[:, w*2:w*3] = segmentation_result
    
    # Add labels
    labels = ['Original', 'Detection Model', 'Segmentation Model']
    positions = [w//2, w + w//2, 2*w + w//2]
    
    for label, pos in zip(labels, positions):
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        x = pos - tw // 2
        # Black background
        cv2.rectangle(comparison, (x - 10, 10), (x + tw + 10, 60), (0, 0, 0), -1)
        cv2.putText(comparison, label, (x, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Add statistics
    stats_y = h - 150
    
    # Detection stats
    det_text = [
        f"Detections: {len(detection_info['detections'])}",
        f"Time: {detection_info['time']:.1f}ms",
        f"Avg Conf: {detection_info['avg_conf']:.2f}" if detection_info['avg_conf'] > 0 else "No detections"
    ]
    y_offset = stats_y
    for text in det_text:
        cv2.putText(comparison, text, (w + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 150), 2)
        y_offset += 30
    
    # Segmentation stats
    seg_text = [
        f"Detections: {len(segmentation_info['detections'])}",
        f"Time: {segmentation_info['time']:.1f}ms",
        f"Avg Conf: {segmentation_info['avg_conf']:.2f}" if segmentation_info['avg_conf'] > 0 else "No detections"
    ]
    y_offset = stats_y
    for text in seg_text:
        cv2.putText(comparison, text, (2*w + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 50), 2)
        y_offset += 30
    
    return comparison


def main():
    """Main comparison function"""
    print("=" * 80)
    print("🔬 MODEL COMPARISON: DETECTION vs SEGMENTATION")
    print("=" * 80)
    
    # Paths
    data_dir = Path('data/upview data')
    detection_model_path = 'model/upview model/detection/detection.pt'
    segmentation_model_path = 'model/upview model/segmentation/segmentation.pt'
    output_dir = 'output_comparison'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/individual', exist_ok=True)
    
    # Load calibration
    calibration = load_calibration()
    if calibration:
        mm_per_pixel = calibration['mm_per_pixel']
        print(f"\n✅ Calibration loaded: {mm_per_pixel:.6f} mm/pixel")
    else:
        mm_per_pixel = None
        print("\n⚠️  No calibration found. Using pixel measurements.")
    
    # Load models
    print(f"\n📦 Loading models...")
    print(f"   Detection model: {detection_model_path}")
    detection_model = YOLO(detection_model_path)
    print(f"   ✅ Detection model loaded (task: {detection_model.task})")
    
    print(f"   Segmentation model: {segmentation_model_path}")
    segmentation_model = YOLO(segmentation_model_path)
    print(f"   ✅ Segmentation model loaded (task: {segmentation_model.task})")
    
    # Get images
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_dir.glob(f'*{ext}'))
        image_files.extend(data_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    print(f"\n📁 Found {len(image_files)} images in {data_dir}")
    
    # Statistics
    detection_stats = {'total_time': 0, 'total_detections': 0, 'total_conf': 0, 'count': 0}
    segmentation_stats = {'total_time': 0, 'total_detections': 0, 'total_conf': 0, 'count': 0}
    
    print(f"\n🔄 Processing images...")
    print("-" * 80)
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        # Load original
        original = cv2.imread(str(img_path))
        
        # Process with detection model
        det_result, det_detections, det_time = process_with_detection_model(
            str(img_path), detection_model, mm_per_pixel
        )
        
        # Process with segmentation model
        seg_result, seg_detections, seg_time = process_with_segmentation_model(
            str(img_path), segmentation_model, mm_per_pixel
        )
        
        # Calculate stats
        det_avg_conf = np.mean([d['confidence'] for d in det_detections]) if det_detections else 0
        seg_avg_conf = np.mean([d['confidence'] for d in seg_detections]) if seg_detections else 0
        
        detection_info = {
            'detections': det_detections,
            'time': det_time,
            'avg_conf': det_avg_conf
        }
        
        segmentation_info = {
            'detections': seg_detections,
            'time': seg_time,
            'avg_conf': seg_avg_conf
        }
        
        # Print results
        print(f"  📊 Detection Model:")
        print(f"     Detections: {len(det_detections)} | Time: {det_time:.1f}ms | Avg Conf: {det_avg_conf:.2f}")
        
        print(f"  📊 Segmentation Model:")
        print(f"     Detections: {len(seg_detections)} | Time: {seg_time:.1f}ms | Avg Conf: {seg_avg_conf:.2f}")
        
        # Update statistics
        detection_stats['total_time'] += det_time
        detection_stats['total_detections'] += len(det_detections)
        detection_stats['total_conf'] += det_avg_conf * len(det_detections) if det_detections else 0
        detection_stats['count'] += 1
        
        segmentation_stats['total_time'] += seg_time
        segmentation_stats['total_detections'] += len(seg_detections)
        segmentation_stats['total_conf'] += seg_avg_conf * len(seg_detections) if seg_detections else 0
        segmentation_stats['count'] += 1
        
        # Create comparison
        comparison = create_comparison_image(
            original, det_result, seg_result,
            detection_info, segmentation_info,
            img_path.name
        )
        
        # Save comparison
        output_path = f'{output_dir}/individual/comparison_{img_path.name}'
        cv2.imwrite(output_path, comparison)
        print(f"  💾 Saved: {output_path}")
    
    # Create summary statistics
    print("\n" + "=" * 80)
    print("📊 OVERALL STATISTICS")
    print("=" * 80)
    
    print(f"\n🟢 DETECTION MODEL:")
    print(f"   Total detections: {detection_stats['total_detections']}")
    print(f"   Average time: {detection_stats['total_time']/detection_stats['count']:.2f}ms")
    avg_det_conf = detection_stats['total_conf'] / detection_stats['total_detections'] if detection_stats['total_detections'] > 0 else 0
    print(f"   Average confidence: {avg_det_conf:.3f}")
    
    print(f"\n🔴 SEGMENTATION MODEL:")
    print(f"   Total detections: {segmentation_stats['total_detections']}")
    print(f"   Average time: {segmentation_stats['total_time']/segmentation_stats['count']:.2f}ms")
    avg_seg_conf = segmentation_stats['total_conf'] / segmentation_stats['total_detections'] if segmentation_stats['total_detections'] > 0 else 0
    print(f"   Average confidence: {avg_seg_conf:.3f}")
    
    # Recommendation
    print(f"\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    
    if avg_seg_conf > avg_det_conf + 0.05:
        print("🏆 WINNER: Segmentation Model")
        print(f"   Reason: Higher confidence ({avg_seg_conf:.3f} vs {avg_det_conf:.3f})")
    elif avg_det_conf > avg_seg_conf + 0.05:
        print("🏆 WINNER: Detection Model")
        print(f"   Reason: Higher confidence ({avg_det_conf:.3f} vs {avg_seg_conf:.3f})")
    else:
        print("🤝 TIE: Both models perform similarly")
        print(f"   Detection: {avg_det_conf:.3f} confidence")
        print(f"   Segmentation: {avg_seg_conf:.3f} confidence")
    
    print(f"\n✅ All comparisons saved to: {output_dir}/individual/")
    print("=" * 80)


if __name__ == '__main__':
    main()

