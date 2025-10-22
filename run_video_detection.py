#!/usr/bin/env python3
"""
YOLO Video Detection with Measurements
Processes video with real-time display and saves output
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
import time
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


def process_video(video_path, model_path, output_dir='output_video', show_realtime=True):
    """
    Process video with YOLO detection and measurements
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        output_dir: Directory to save output video
        show_realtime: Whether to show real-time detection window
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
        print("⚠️  No calibration found. Showing pixel measurements only.")
    
    # Load YOLO model (supports protected models)
    print(f"\n🤖 Loading model: {model_path}")
    if 'protected_' in model_path or os.path.exists(model_path + '.lock'):
        model = load_protected_model(model_path)
    else:
        model = YOLO(model_path)
    
    # Open video
    print(f"📹 Opening video: {video_path}")
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
    output_path = os.path.join(output_dir, f'detected_{video_name}.mp4')
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
    
    # Create window for real-time display
    if show_realtime:
        cv2.namedWindow('YOLO Detection - Press Q to quit', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Detection - Press Q to quit', 1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        det_start = time.time()
        results = model(frame, verbose=False)
        det_time = (time.time() - det_start) * 1000
        detection_times.append(det_time)
        
        # Process detections
        frame_overlay = frame.copy()
        detection_count = 0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    detection_count += 1
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    # Calculate dimensions
                    width_px = x2 - x1
                    height_px = y2 - y1
                    
                    # Generate color (consistent per class)
                    color = (int(cls_id * 50 % 255), int(cls_id * 100 % 255), int(cls_id * 150 % 255))
                    if sum(color) < 200:
                        color = (50, 255, 150)
                    
                    # Draw semi-transparent filled rectangle
                    cv2.rectangle(frame_overlay, (x1, y1), (x2, y2), color, -1)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Calculate measurements if calibration available
                    if mm_per_pixel:
                        height_mm = height_px * mm_per_pixel
                        
                        # Draw vertical measurement line
                        mid_x = (x1 + x2) // 2
                        cv2.line(frame, (mid_x, y1), (mid_x, y2), (0, 255, 255), 2)
                        cv2.circle(frame, (mid_x, y1), 5, (0, 255, 255), -1)
                        cv2.circle(frame, (mid_x, y2), 5, (0, 255, 255), -1)
                        
                        # Add measurement text
                        height_text = f"{height_mm:.1f}mm"
                        (tw, th), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        text_x = mid_x + 15
                        text_y = (y1 + y2) // 2
                        
                        # Background for measurement
                        cv2.rectangle(frame, (text_x - 5, text_y - th - 5),
                                    (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
                        cv2.putText(frame, height_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        measurement_text = f"{height_mm:.1f}mm"
                    else:
                        measurement_text = f"{height_px}px"
                    
                    # Add class label
                    label = f"{cls_name} {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - lh - 15), (x1 + lw + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blend overlay
        frame = cv2.addWeighted(frame, 0.7, frame_overlay, 0.3, 0)
        
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
            f"Detections: {detection_count} | FPS: {current_fps:.1f}",
            f"Detection time: {det_time:.1f}ms"
        ]
        
        y_offset = 25
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Write frame to output video
        out.write(frame)
        
        # Show real-time display
        if show_realtime:
            cv2.imshow('YOLO Detection - Press Q to quit', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⚠️  User interrupted. Saving processed frames...")
                break
        
        # Print progress every 30 frames
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
    
    # Print summary
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    avg_det = np.mean(detection_times)
    
    print("\n" + "=" * 60)
    print("✅ VIDEO PROCESSING COMPLETE")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"   Total frames processed: {frame_count}/{total_frames}")
    print(f"   Processing time: {elapsed_time:.2f} seconds")
    print(f"   Average FPS: {avg_fps:.2f}")
    print(f"   Average detection time: {avg_det:.2f}ms")
    print(f"   Output saved to: {output_path}")
    print("=" * 60)


def main():
    """Main function"""
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
    video_files = []
    
    # Check data folder and subfolders
    data_paths = ['data', 'data/video']
    for data_path in data_paths:
        if os.path.exists(data_path):
            for ext in video_extensions:
                video_files.extend(Path(data_path).glob(f'*{ext}'))
    
    if not video_files:
        print("❌ No video files found in data/ or data/video/ folder")
        print("   Supported formats: .mp4, .avi, .mov, .mkv")
        return
    
    print("=" * 60)
    print("🎬 YOLO VIDEO DETECTION WITH MEASUREMENTS")
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
                else:
                    print(f"❌ Please enter a number between 1 and {len(video_files)}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    
    # Check model
    model_path = 'model/my_model.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    print("\n" + "=" * 60)
    
    # Process video
    process_video(
        video_path=str(selected_video),
        model_path=model_path,
        output_dir='output_video',
        show_realtime=True
    )


if __name__ == '__main__':
    main()

