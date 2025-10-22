#!/usr/bin/env python3
"""
Upview Camera Calibration Tool
Uses detection model to get bounding box for calibration
Saves separate calibration file for upview camera
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO


class UpviewCalibrationTool:
    def __init__(self, image_path, model_path):
        self.image_path = image_path
        self.model_path = model_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_image = self.image.copy()
        self.detections = []
        self.selected_detection = None
        self.window_name = "Upview Calibration - Select Detection"
        
    def run_detection(self):
        """Run YOLO detection to find bounding boxes"""
        print("\n🤖 Running YOLO detection...")
        model = YOLO(self.model_path)
        results = model(self.image_path)
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    height_px = y2 - y1
                    width_px = x2 - x1
                    
                    self.detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': cls_name,
                        'confidence': conf,
                        'height_px': height_px,
                        'width_px': width_px,
                        'index': i
                    })
        
        print(f"✅ Found {len(self.detections)} detection(s)")
        return len(self.detections) > 0
    
    def draw_detections(self):
        """Draw all detected bounding boxes"""
        self.display_image = self.image.copy()
        
        for i, det in enumerate(self.detections):
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, 3)
            
            # Draw index number
            cv2.circle(self.display_image, (x1 + 15, y1 + 15), 18, color, -1)
            cv2.putText(self.display_image, str(i + 1), (x1 + 8, y1 + 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(self.display_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw height measurement line
            mid_x = (x1 + x2) // 2
            cv2.line(self.display_image, (mid_x, y1), (mid_x, y2), (255, 255, 0), 3)
            cv2.circle(self.display_image, (mid_x, y1), 6, (255, 255, 0), -1)
            cv2.circle(self.display_image, (mid_x, y2), 6, (255, 255, 0), -1)
            
            # Show pixel height
            height_text = f"{det['height_px']}px"
            text_x = mid_x + 15
            text_y = (y1 + y2) // 2
            cv2.putText(self.display_image, height_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    def select_detection(self):
        """Let user select which detection to use for calibration"""
        if len(self.detections) == 0:
            print("❌ No detections found!")
            return None
        
        self.draw_detections()
        
        print("\n" + "=" * 60)
        print("📦 DETECTED OBJECTS FOR UPVIEW CALIBRATION:")
        print("=" * 60)
        for i, det in enumerate(self.detections):
            print(f"{i + 1}. {det['class']} (confidence: {det['confidence']:.2f})")
            print(f"   Vertical Height: {det['height_px']} pixels")
            print(f"   Horizontal Width: {det['width_px']} pixels")
            print(f"   Bounding Box: {det['bbox']}")
        print("=" * 60)
        
        # Show image
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 900)
        cv2.imshow(self.window_name, self.display_image)
        
        # Get user selection
        while True:
            try:
                if len(self.detections) == 1:
                    print(f"\n✅ Only one detection found, using it automatically.")
                    choice = 1
                else:
                    choice = input(f"\nSelect detection number (1-{len(self.detections)}): ").strip()
                    choice = int(choice)
                
                if 1 <= choice <= len(self.detections):
                    self.selected_detection = self.detections[choice - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(self.detections)}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
        
        cv2.destroyAllWindows()
        return self.selected_detection
    
    def get_real_measurement(self):
        """Get real-world measurement from user"""
        det = self.selected_detection
        
        print("\n" + "=" * 60)
        print("📏 UPVIEW CALIBRATION SETUP")
        print("=" * 60)
        print(f"Selected: {det['class']} (confidence: {det['confidence']:.2f})")
        print(f"Bounding box VERTICAL height: {det['height_px']} pixels")
        print("=" * 60)
        
        print("\n⚠️  IMPORTANT:")
        print("   Measure the ACTUAL VERTICAL HEIGHT of this object")
        print("   in real life using a ruler or caliper.")
        print("   This is the upview camera - measure top-to-bottom.")
        print("=" * 60)
        
        # Get real-world measurement
        while True:
            try:
                mm_distance = float(input("\n📏 Enter the VERTICAL HEIGHT in millimeters (mm): "))
                if mm_distance <= 0:
                    print("❌ Distance must be positive. Try again.")
                    continue
                
                # Confirm
                print(f"\n✅ You entered: {mm_distance} mm")
                confirm = input("   Is this correct? (y/n): ").strip().lower()
                if confirm == 'y' or confirm == 'yes':
                    break
                else:
                    print("   Let's try again...")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
        
        return mm_distance
    
    def calculate_calibration(self, mm_distance):
        """Calculate calibration from bounding box"""
        det = self.selected_detection
        pixel_distance = det['height_px']
        
        # Calculate calibration factor
        mm_per_pixel = mm_distance / pixel_distance
        
        calibration_data = {
            'pixel_distance': float(pixel_distance),
            'mm_distance': float(mm_distance),
            'mm_per_pixel': float(mm_per_pixel),
            'bbox': det['bbox'],
            'class': det['class'],
            'confidence': float(det['confidence']),
            'calibration_image': str(self.image_path),
            'camera': 'upview',
            'method': 'bounding_box'
        }
        
        print("\n" + "=" * 60)
        print("✅ UPVIEW CALIBRATION COMPLETE")
        print("=" * 60)
        print(f"📦 Detection: {det['class']} ({det['confidence']:.2f})")
        print(f"📏 Bounding box height: {pixel_distance} pixels")
        print(f"📏 Real height: {mm_distance:.2f} mm")
        print(f"🎯 Calibration factor: {mm_per_pixel:.6f} mm/pixel")
        print(f"📹 Camera: UPVIEW")
        print("=" * 60)
        
        return calibration_data
    
    @staticmethod
    def save_calibration(calibration_data, filename='calibration_upview.json'):
        """Save calibration to file"""
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"\n💾 Upview calibration saved to: {filename}")
        return filename


def main():
    """Main calibration function"""
    
    # Model path
    model_path = 'model/upview model/detection/detection.pt'
    
    # Check model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Get all images in upview data folder
    data_dir = Path('data/upview data')
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_dir.glob(f'*{ext}'))
        image_files.extend(data_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print("❌ No images found in data/upview data/ folder")
        return
    
    print("=" * 60)
    print("📏 UPVIEW CAMERA CALIBRATION TOOL")
    print("=" * 60)
    print("\nThis tool calibrates the UPVIEW camera separately.")
    print("Uses the detection model for accurate bounding boxes.\n")
    
    print("📁 Available images for calibration:")
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {img.name}")
    
    # Select image
    while True:
        try:
            choice = input(f"\nSelect image (1-{len(image_files)}) or press Enter for first image: ").strip()
            if choice == "":
                selected_image = image_files[0]
                break
            idx = int(choice) - 1
            if 0 <= idx < len(image_files):
                selected_image = image_files[idx]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(image_files)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    print(f"\n✅ Using image: {selected_image.name}")
    
    # Run calibration
    try:
        tool = UpviewCalibrationTool(str(selected_image), model_path)
        
        # Run detection
        if not tool.run_detection():
            print("❌ No objects detected in this image. Please try another image.")
            return
        
        # Select detection
        if tool.select_detection() is None:
            print("❌ No detection selected.")
            return
        
        # Get real measurement
        mm_distance = tool.get_real_measurement()
        
        # Calculate calibration
        calibration_data = tool.calculate_calibration(mm_distance)
        
        # Save calibration
        UpviewCalibrationTool.save_calibration(calibration_data)
        
        print("\n✅ Upview calibration complete!")
        print("   You can now run upview detection with measurements:")
        print("   python run_upview_detection.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

