#!/usr/bin/env python3
"""
Calibration Tool - Click two points to set pixel-to-mm ratio
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

class CalibrationTool:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_image = self.image.copy()
        self.points = []
        self.window_name = "Calibration Tool - Click 2 Points"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                
                # Draw the point
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.display_image, f"P{len(self.points)}", 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)
                
                # If we have 2 points, draw the line
                if len(self.points) == 2:
                    cv2.line(self.display_image, self.points[0], self.points[1], 
                            (0, 255, 0), 2)
                    
                    # Calculate pixel distance
                    dx = self.points[1][0] - self.points[0][0]
                    dy = self.points[1][1] - self.points[0][1]
                    pixel_distance = np.sqrt(dx**2 + dy**2)
                    
                    # Display pixel distance
                    mid_x = (self.points[0][0] + self.points[1][0]) // 2
                    mid_y = (self.points[0][1] + self.points[1][1]) // 2
                    cv2.putText(self.display_image, f"{pixel_distance:.1f} pixels", 
                               (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 0), 2)
                
                cv2.imshow(self.window_name, self.display_image)
    
    def run(self):
        """Run the calibration tool"""
        print("=" * 60)
        print("📏 CALIBRATION TOOL")
        print("=" * 60)
        print("\nInstructions:")
        print("1. Click on the FIRST point of a known distance")
        print("2. Click on the SECOND point of the same known distance")
        print("3. The tool will show the pixel distance")
        print("4. Enter the real-world measurement in mm")
        print("5. Press 'r' to reset and start over")
        print("6. Press 'q' to quit without saving")
        print("\n" + "=" * 60)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.imshow(self.window_name, self.display_image)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Reset
            if key == ord('r'):
                print("\n🔄 Resetting points...")
                self.points = []
                self.display_image = self.image.copy()
                cv2.imshow(self.window_name, self.display_image)
            
            # Quit
            elif key == ord('q'):
                print("\n❌ Calibration cancelled.")
                cv2.destroyAllWindows()
                return None
            
            # If we have 2 points, wait for measurement input
            if len(self.points) == 2:
                cv2.destroyAllWindows()
                return self.get_calibration()
        
    def get_calibration(self):
        """Calculate and save calibration"""
        # Calculate pixel distance
        dx = self.points[1][0] - self.points[0][0]
        dy = self.points[1][1] - self.points[0][1]
        pixel_distance = np.sqrt(dx**2 + dy**2)
        
        print(f"\n✅ Two points selected!")
        print(f"📐 Pixel distance: {pixel_distance:.2f} pixels")
        print(f"   Point 1: {self.points[0]}")
        print(f"   Point 2: {self.points[1]}")
        
        # Get real-world measurement
        while True:
            try:
                mm_distance = float(input("\n📏 Enter the real-world distance in millimeters (mm): "))
                if mm_distance <= 0:
                    print("❌ Distance must be positive. Try again.")
                    continue
                break
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
        
        # Calculate calibration factor (mm per pixel)
        mm_per_pixel = mm_distance / pixel_distance
        
        calibration_data = {
            'pixel_distance': pixel_distance,
            'mm_distance': mm_distance,
            'mm_per_pixel': mm_per_pixel,
            'point1': self.points[0],
            'point2': self.points[1],
            'calibration_image': str(self.image_path)
        }
        
        print("\n" + "=" * 60)
        print("✅ CALIBRATION COMPLETE")
        print("=" * 60)
        print(f"📏 Pixel distance: {pixel_distance:.2f} pixels")
        print(f"📏 Real distance: {mm_distance:.2f} mm")
        print(f"🎯 Calibration factor: {mm_per_pixel:.4f} mm/pixel")
        print("=" * 60)
        
        return calibration_data
    
    @staticmethod
    def save_calibration(calibration_data, filename='calibration.json'):
        """Save calibration to file"""
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"\n💾 Calibration saved to: {filename}")
        return filename
    
    @staticmethod
    def load_calibration(filename='calibration.json'):
        """Load calibration from file"""
        if not os.path.exists(filename):
            return None
        with open(filename, 'r') as f:
            return json.load(f)


def main():
    """Main calibration function"""
    # Get all images in data folder
    data_dir = Path('data')
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_dir.glob(f'*{ext}'))
        image_files.extend(data_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print("❌ No images found in data/ folder")
        return
    
    print("\n📁 Available images for calibration:")
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
    tool = CalibrationTool(str(selected_image))
    calibration_data = tool.run()
    
    if calibration_data:
        # Save calibration
        CalibrationTool.save_calibration(calibration_data)
        
        print("\n✅ You can now run the segmentation script with measurements!")
        print("   Run: python run_segmentation.py")
    else:
        print("\n❌ Calibration was not completed.")


if __name__ == '__main__':
    main()

