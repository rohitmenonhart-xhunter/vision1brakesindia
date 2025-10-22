#!/usr/bin/env python3
"""
Upview Segmentation Model - Video Processing
Process videos with pixel-perfect segmentation masks
Real-time display with measurements
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
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


def process_video(video_path, model_path, output_dir='output_upview_video', show_realtime=True):
    """Process video with segmentation model"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load calibration
    calibration = load_calibration()
    if calibration:
        mm_per_pixel = calibration['mm_per_pixel']
        print(f"✅ Calibration loaded: {mm_per_pixel:.6f} mm/pixel")
    else:
        mm_per_pixel = None
        print("⚠️  No calibration found. Using pixel measurements.")
    
    # Load model
    print(f"\n🤖 Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"   Model task: {model.task}")
    
    # Open video
    print(f"\n📹 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f} seconds")
    
    # Setup video writer
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f'segmented_{video_name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n💾 Output will be saved to: {output_path}")
    print(f"🎬 Processing video...")
    if show_realtime:
        print(f"👁️  Real-time display enabled (Press 'q' to quit)")
    print("-" * 60)
    
    # Process video
    frame_count = 0
    start_time = time.time()
    detection_times = []
    
    # Create window
    if show_realtime:
        cv2.namedWindow('Upview Segmentation - Press Q to quit', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Upview Segmentation - Press Q to quit', 1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run segmentation
        det_start = time.time()
        results = model(frame, verbose=False)
        det_time = (time.time() - det_start) * 1000
        detection_times.append(det_time)
        
        # Process results
        frame_overlay = frame.copy()
        detection_count = 0
        
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                for idx, (mask, box) in enumerate(zip(result.masks, result.boxes)):
                    detection_count += 1
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    # Calculate dimensions
                    height_px = y2 - y1
                    
                    # Get mask
                    mask_array = mask.data[0].cpu().numpy()
                    mask_resized = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]))
                    mask_bool = mask_resized > 0.5
                    
                    # Color (unique per instance)
                    np.random.seed(idx + 42)
                    color = tuple(map(int, np.random.randint(50, 255, 3)))
                    
                    # Apply mask
                    frame_overlay[mask_bool] = color
                    
                    # Draw contour
                    contours, _ = cv2.findContours(mask_bool.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, color, 3)
                    
                    # Add measurements
                    if mm_per_pixel:
                        height_mm = height_px * mm_per_pixel
                        
                        # Draw measurement line
                        mid_x = (x1 + x2) // 2
                        cv2.line(frame, (mid_x, y1), (mid_x, y2), (0, 255, 255), 2)
                        cv2.circle(frame, (mid_x, y1), 5, (0, 255, 255), -1)
                        cv2.circle(frame, (mid_x, y2), 5, (0, 255, 255), -1)
                        
                        # Measurement text
                        height_text = f"{height_mm:.1f}mm"
                        (tw, th), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        text_x = mid_x + 15
                        text_y = (y1 + y2) // 2
                        
                        cv2.rectangle(frame, (text_x - 5, text_y - th - 5),
                                     (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
                        cv2.putText(frame, height_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Add class label
                    label = f"{cls_name} {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - lh - 15), (x1 + lw + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blend overlay
        frame = cv2.addWeighted(frame, 0.6, frame_overlay, 0.4, 0)
        
        # Add info overlay
        info_bg_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, info_bg_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add frame info
        progress = (frame_count / total_frames) * 100
        avg_det_time = np.mean(detection_times[-30:]) if detection_times else 0
        current_fps = 1000 / avg_det_time if avg_det_time > 0 else 0
        
        info_lines = [
            f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)",
            f"Segmentations: {detection_count} | FPS: {current_fps:.1f}",
            f"Detection time: {det_time:.1f}ms"
        ]
        
        y_offset = 25
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Write frame
        out.write(frame)
        
        # Show real-time
        if show_realtime:
            cv2.imshow('Upview Segmentation - Press Q to quit', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⚠️  User interrupted. Saving processed frames...")
                break
        
        # Progress updates
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
            print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"Speed: {fps_actual:.1f} fps | ETA: {eta:.1f}s")
    
    # Cleanup
    cap.release()
    out.release()
    if show_realtime:
        cv2.destroyAllWindows()
    
    # Summary
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    avg_det = np.mean(detection_times)
    
    print("\n" + "=" * 60)
    print("✅ VIDEO PROCESSING COMPLETE")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"   Frames processed: {frame_count}/{total_frames}")
    print(f"   Processing time: {elapsed_time:.2f} seconds")
    print(f"   Average FPS: {avg_fps:.2f}")
    print(f"   Average detection time: {avg_det:.2f}ms")
    print(f"   Output: {output_path}")
    print("=" * 60)


def main():
    """Main function"""
    # Find videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    data_paths = ['data', 'data/video', 'data/upview data']
    for data_path in data_paths:
        if os.path.exists(data_path):
            for ext in video_extensions:
                video_files.extend(Path(data_path).glob(f'*{ext}'))
                video_files.extend(Path(data_path).glob(f'*{ext.upper()}'))
    
    if not video_files:
        print("❌ No video files found")
        return
    
    print("=" * 60)
    print("🎬 UPVIEW SEGMENTATION - VIDEO PROCESSING")
    print("=" * 60)
    
    print("\n📹 Available videos:")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {video}")
    
    # Select video
    if len(video_files) == 1:
        selected_video = video_files[0]
        print(f"\n✅ Using: {selected_video.name}")
    else:
        while True:
            try:
                choice = input(f"\nSelect video (1-{len(video_files)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(video_files):
                    selected_video = video_files[idx]
                    break
                print(f"❌ Please enter 1-{len(video_files)}")
            except ValueError:
                print("❌ Invalid input")
    
    # Check model
    model_path = 'model/upview model/segmentation/segmentation.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print("\n" + "=" * 60)
    
    # Process
    process_video(
        video_path=str(selected_video),
        model_path=model_path,
        output_dir='output_upview_video',
        show_realtime=True
    )


if __name__ == '__main__':
    main()

