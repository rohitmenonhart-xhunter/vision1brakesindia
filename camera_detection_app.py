#!/usr/bin/env python3
"""
Interactive Camera Detection Application
Features:
- Live camera feed
- First capture: Calibrate distance
- Subsequent captures: Detect objects and show measurements
- GUI-based interface
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
from datetime import datetime
from secure_loader import load_protected_model


class CameraDetectionApp:
    def __init__(self, root, model_path='model/protected_my_model.pt'):
        self.root = root
        self.root.title("🎥 Live Camera Detection & Measurement")
        self.root.geometry("1400x900")
        
        # State variables
        self.model_path = model_path
        self.model = None
        self.camera = None
        self.camera_index = 0
        self.is_running = False
        self.calibration = None
        self.mm_per_pixel = None
        self.calibration_mode = False
        self.calibration_points = []
        self.current_frame = None
        self.captured_frame = None
        self.photo_count = 0
        
        # Create GUI
        self.setup_gui()
        
        # Load model
        self.load_model()
        
        # Load existing calibration if available
        self.load_calibration()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        
        # ===== TOP CONTROL PANEL =====
        control_frame = tk.Frame(self.root, bg='#2c3e50', pady=10)
        control_frame.pack(fill=tk.X)
        
        # Title
        title_label = tk.Label(control_frame, text="🎥 Camera Detection System", 
                              font=('Arial', 20, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=5)
        
        # Button panel
        button_frame = tk.Frame(control_frame, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        # Start/Stop Camera Button
        self.btn_camera = tk.Button(button_frame, text="📹 Start Camera", 
                                    command=self.toggle_camera,
                                    font=('Arial', 12, 'bold'),
                                    bg='#27ae60', fg='white',
                                    padx=20, pady=10)
        self.btn_camera.pack(side=tk.LEFT, padx=5)
        
        # Capture Photo Button
        self.btn_capture = tk.Button(button_frame, text="📸 Capture Photo", 
                                     command=self.capture_photo,
                                     font=('Arial', 12, 'bold'),
                                     bg='#3498db', fg='white',
                                     padx=20, pady=10,
                                     state=tk.DISABLED)
        self.btn_capture.pack(side=tk.LEFT, padx=5)
        
        # Recalibrate Button
        self.btn_recalibrate = tk.Button(button_frame, text="🔧 Recalibrate", 
                                         command=self.start_calibration,
                                         font=('Arial', 12, 'bold'),
                                         bg='#e67e22', fg='white',
                                         padx=20, pady=10)
        self.btn_recalibrate.pack(side=tk.LEFT, padx=5)
        
        # Clear Results Button
        self.btn_clear = tk.Button(button_frame, text="🗑️ Clear Results", 
                                   command=self.clear_results,
                                   font=('Arial', 12, 'bold'),
                                   bg='#e74c3c', fg='white',
                                   padx=20, pady=10)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        # ===== STATUS BAR =====
        status_frame = tk.Frame(self.root, bg='#34495e', pady=5)
        status_frame.pack(fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="📊 Status: Ready", 
                                     font=('Arial', 11), bg='#34495e', fg='white')
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.calibration_label = tk.Label(status_frame, text="⚠️ Not Calibrated", 
                                         font=('Arial', 11), bg='#34495e', fg='#e74c3c')
        self.calibration_label.pack(side=tk.LEFT, padx=20)
        
        self.photo_label = tk.Label(status_frame, text="📸 Photos: 0", 
                                   font=('Arial', 11), bg='#34495e', fg='white')
        self.photo_label.pack(side=tk.LEFT, padx=20)
        
        # ===== MAIN CONTENT AREA =====
        content_frame = tk.Frame(self.root, bg='#ecf0f1')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Camera Feed
        left_frame = tk.LabelFrame(content_frame, text="📹 Live Camera Feed", 
                                   font=('Arial', 12, 'bold'),
                                   bg='#ecf0f1', padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.camera_label = tk.Label(left_frame, bg='black', 
                                     text="Camera Off\n\nClick 'Start Camera' to begin",
                                     font=('Arial', 14), fg='white')
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Results Panel
        right_frame = tk.LabelFrame(content_frame, text="📊 Detection Results", 
                                    font=('Arial', 12, 'bold'),
                                    bg='#ecf0f1', padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Results scrollable area
        results_scroll = tk.Scrollbar(right_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(right_frame, font=('Courier', 10),
                                   bg='#2c3e50', fg='#ecf0f1',
                                   yscrollcommand=results_scroll.set,
                                   wrap=tk.WORD, padx=10, pady=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.results_text.yview)
        
        # Initial message
        self.add_result("🎯 Camera Detection System Ready!\n")
        self.add_result("=" * 50)
        self.add_result("\n📋 Instructions:\n")
        self.add_result("1. Click 'Start Camera' to begin\n")
        self.add_result("2. First photo: Calibrate distance\n")
        self.add_result("3. Next photos: Auto-detect & measure\n")
        self.add_result("4. Use 'Recalibrate' to reset calibration\n")
        self.add_result("=" * 50 + "\n\n")
        
    def load_model(self):
        """Load YOLO model"""
        try:
            self.add_result(f"🤖 Loading model: {self.model_path}\n")
            
            # Check if it's a protected model
            if 'protected_' in self.model_path or os.path.exists(self.model_path + '.lock'):
                self.model = load_protected_model(self.model_path)
            else:
                self.model = YOLO(self.model_path)
            
            self.add_result(f"✅ Model loaded successfully!\n")
            self.add_result(f"   Task: {self.model.task}\n\n")
            
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model:\n{str(e)}")
            self.add_result(f"❌ Model loading failed: {str(e)}\n\n")
    
    def load_calibration(self):
        """Load existing calibration if available"""
        if os.path.exists('calibration_camera.json'):
            try:
                with open('calibration_camera.json', 'r') as f:
                    self.calibration = json.load(f)
                self.mm_per_pixel = self.calibration['mm_per_pixel']
                
                self.calibration_label.config(
                    text=f"✅ Calibrated: {self.mm_per_pixel:.4f} mm/px",
                    fg='#27ae60'
                )
                self.add_result(f"✅ Loaded calibration: {self.mm_per_pixel:.4f} mm/pixel\n")
                self.add_result(f"   Based on: {self.calibration['mm_distance']:.2f}mm = ")
                self.add_result(f"{self.calibration['pixel_distance']:.2f}px\n\n")
            except:
                pass
    
    def toggle_camera(self):
        """Start or stop camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera feed"""
        self.camera = cv2.VideoCapture(self.camera_index)
        
        if not self.camera.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera!")
            return
        
        # Set camera resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.is_running = True
        self.btn_camera.config(text="⏹️ Stop Camera", bg='#e74c3c')
        self.btn_capture.config(state=tk.NORMAL)
        self.status_label.config(text="📊 Status: Camera Active")
        
        self.add_result("📹 Camera started!\n\n")
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.update_camera_feed, daemon=True)
        self.camera_thread.start()
    
    def stop_camera(self):
        """Stop camera feed"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        self.btn_camera.config(text="📹 Start Camera", bg='#27ae60')
        self.btn_capture.config(state=tk.DISABLED)
        self.status_label.config(text="📊 Status: Camera Stopped")
        
        # Clear camera display
        self.camera_label.config(image='', 
                                text="Camera Off\n\nClick 'Start Camera' to begin")
        
        self.add_result("⏹️ Camera stopped.\n\n")
    
    def update_camera_feed(self):
        """Update camera feed continuously"""
        while self.is_running:
            ret, frame = self.camera.read()
            
            if ret:
                self.current_frame = frame.copy()
                
                # If in calibration mode, show calibration points
                if self.calibration_mode:
                    for i, point in enumerate(self.calibration_points):
                        cv2.circle(frame, point, 5, (0, 255, 0), -1)
                        cv2.putText(frame, f"Point {i+1}", 
                                  (point[0] + 10, point[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if len(self.calibration_points) == 2:
                        cv2.line(frame, self.calibration_points[0], 
                               self.calibration_points[1], (0, 255, 255), 2)
                
                # Convert for Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (800, 600))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk, text='')
    
    def capture_photo(self):
        """Capture photo from camera"""
        if self.current_frame is None:
            messagebox.showwarning("Capture Error", "No camera frame available!")
            return
        
        self.captured_frame = self.current_frame.copy()
        self.photo_count += 1
        self.photo_label.config(text=f"📸 Photos: {self.photo_count}")
        
        # Save captured photo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"captures/capture_{timestamp}.jpg"
        os.makedirs("captures", exist_ok=True)
        cv2.imwrite(save_path, self.captured_frame)
        
        self.add_result(f"📸 Photo {self.photo_count} captured! Saved to: {save_path}\n")
        
        # If not calibrated, start calibration
        if self.mm_per_pixel is None:
            self.add_result("⚠️  No calibration found. Starting calibration...\n\n")
            self.start_calibration_on_frame()
        else:
            # Run detection
            self.run_detection()
    
    def start_calibration(self):
        """Start calibration process - capture new photo first"""
        if not self.is_running:
            messagebox.showwarning("Camera Error", "Please start camera first!")
            return
        
        response = messagebox.askyesno("Calibration", 
                                       "This will capture a new photo for calibration.\n\n"
                                       "Make sure a reference object is visible in the frame.\n\n"
                                       "Continue?")
        if response:
            self.captured_frame = self.current_frame.copy()
            self.photo_count += 1
            self.photo_label.config(text=f"📸 Photos: {self.photo_count}")
            self.add_result(f"\n📸 Calibration photo captured!\n")
            self.start_calibration_on_frame()
    
    def start_calibration_on_frame(self):
        """Start calibration on captured frame"""
        if self.captured_frame is None:
            return
        
        self.calibration_mode = True
        self.calibration_points = []
        
        # Create calibration window
        cal_window = tk.Toplevel(self.root)
        cal_window.title("🔧 Calibration - Click Two Points")
        cal_window.geometry("900x700")
        
        instruction = tk.Label(cal_window, 
                              text="Click two points on the reference object\n"
                                   "Then enter the distance between them in mm",
                              font=('Arial', 12), bg='#f39c12', fg='white', pady=10)
        instruction.pack(fill=tk.X)
        
        # Display image
        cal_canvas = tk.Canvas(cal_window, bg='black')
        cal_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Convert and display image
        display_frame = self.captured_frame.copy()
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (850, 600))
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)
        
        cal_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        cal_canvas.image = img_tk
        
        # Calculate scale factors
        scale_x = self.captured_frame.shape[1] / 850
        scale_y = self.captured_frame.shape[0] / 600
        
        def on_click(event):
            if len(self.calibration_points) < 2:
                # Scale coordinates back to original image size
                orig_x = int(event.x * scale_x)
                orig_y = int(event.y * scale_y)
                self.calibration_points.append((orig_x, orig_y))
                
                # Draw point on canvas
                cal_canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, 
                                      fill='lime', outline='lime')
                cal_canvas.create_text(event.x+20, event.y-10, 
                                      text=f"Point {len(self.calibration_points)}",
                                      fill='lime', font=('Arial', 10, 'bold'))
                
                if len(self.calibration_points) == 2:
                    # Draw line
                    p1_canvas = (int(self.calibration_points[0][0] / scale_x),
                               int(self.calibration_points[0][1] / scale_y))
                    p2_canvas = (int(self.calibration_points[1][0] / scale_x),
                               int(self.calibration_points[1][1] / scale_y))
                    cal_canvas.create_line(p1_canvas[0], p1_canvas[1],
                                         p2_canvas[0], p2_canvas[1],
                                         fill='yellow', width=2)
                    
                    # Calculate pixel distance
                    pixel_dist = np.sqrt(
                        (self.calibration_points[1][0] - self.calibration_points[0][0])**2 +
                        (self.calibration_points[1][1] - self.calibration_points[0][1])**2
                    )
                    
                    # Ask for real distance
                    cal_window.withdraw()
                    mm_distance = simpledialog.askfloat("Enter Distance",
                                                       f"Pixel distance: {pixel_dist:.1f} px\n\n"
                                                       f"Enter the REAL distance in millimeters:",
                                                       parent=self.root)
                    
                    if mm_distance and mm_distance > 0:
                        # Calculate calibration
                        self.mm_per_pixel = mm_distance / pixel_dist
                        
                        # Save calibration
                        self.calibration = {
                            'pixel_distance': float(pixel_dist),
                            'mm_distance': float(mm_distance),
                            'mm_per_pixel': float(self.mm_per_pixel),
                            'camera': 'webcam'
                        }
                        
                        with open('calibration_camera.json', 'w') as f:
                            json.dump(self.calibration, f, indent=2)
                        
                        self.calibration_label.config(
                            text=f"✅ Calibrated: {self.mm_per_pixel:.4f} mm/px",
                            fg='#27ae60'
                        )
                        
                        self.add_result(f"\n✅ Calibration complete!\n")
                        self.add_result(f"   Pixel distance: {pixel_dist:.2f} px\n")
                        self.add_result(f"   Real distance: {mm_distance:.2f} mm\n")
                        self.add_result(f"   Calibration: {self.mm_per_pixel:.4f} mm/pixel\n")
                        self.add_result(f"   Saved to: calibration_camera.json\n\n")
                        
                        messagebox.showinfo("Success", 
                                          f"Calibration successful!\n\n"
                                          f"{self.mm_per_pixel:.4f} mm/pixel")
                    else:
                        self.add_result("❌ Calibration cancelled.\n\n")
                    
                    self.calibration_mode = False
                    cal_window.destroy()
        
        cal_canvas.bind("<Button-1>", on_click)
    
    def run_detection(self):
        """Run detection on captured frame"""
        if self.captured_frame is None or self.model is None:
            return
        
        self.add_result("🔍 Running detection...\n")
        
        # Run inference
        results = self.model(self.captured_frame, verbose=False)
        
        # Process results
        result_frame = self.captured_frame.copy()
        detections_found = 0
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = self.model.names[cls]
                
                detections_found += 1
                
                # Calculate dimensions
                width_px = x2 - x1
                height_px = y2 - y1
                
                # Draw bounding box
                color = (0, 255, 0)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add measurements if calibrated
                if self.mm_per_pixel:
                    height_mm = height_px * self.mm_per_pixel
                    width_mm = width_px * self.mm_per_pixel
                    
                    # Draw vertical measurement line
                    mid_x = (x1 + x2) // 2
                    cv2.line(result_frame, (mid_x, y1), (mid_x, y2), (0, 255, 255), 2)
                    cv2.circle(result_frame, (mid_x, y1), 4, (0, 255, 255), -1)
                    cv2.circle(result_frame, (mid_x, y2), 4, (0, 255, 255), -1)
                    
                    # Measurement text
                    height_text = f"{height_mm:.1f}mm"
                    (tw, th), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = mid_x + 10
                    text_y = (y1 + y2) // 2
                    
                    cv2.rectangle(result_frame, (text_x - 5, text_y - th - 5),
                                (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
                    cv2.putText(result_frame, height_text, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Class label
                    label = f"{cls_name} {conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(result_frame, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
                    cv2.putText(result_frame, label, (x1 + 5, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Log to results
                    self.add_result(f"  ✅ {cls_name} (conf: {conf:.2f})\n")
                    self.add_result(f"     Height: {height_mm:.2f} mm ({height_px} px)\n")
                    self.add_result(f"     Width: {width_mm:.2f} mm ({width_px} px)\n")
                else:
                    # No calibration - just show pixels
                    label = f"{cls_name} {conf:.2f} | {height_px}x{width_px}px"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(result_frame, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
                    cv2.putText(result_frame, label, (x1 + 5, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    self.add_result(f"  ✅ {cls_name} (conf: {conf:.2f})\n")
                    self.add_result(f"     Size: {height_px}x{width_px} px\n")
        
        if detections_found == 0:
            self.add_result("  ⚠️  No objects detected.\n")
        else:
            self.add_result(f"\n📊 Total detections: {detections_found}\n")
        
        self.add_result("-" * 50 + "\n\n")
        
        # Save result image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"captures/result_{timestamp}.jpg"
        cv2.imwrite(result_path, result_frame)
        self.add_result(f"💾 Result saved to: {result_path}\n\n")
        
        # Display result in new window
        self.show_result_window(result_frame)
    
    def show_result_window(self, result_frame):
        """Show detection result in new window"""
        result_window = tk.Toplevel(self.root)
        result_window.title("Detection Result")
        result_window.geometry("1000x750")
        
        # Convert and display
        frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (950, 700))
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)
        
        label = tk.Label(result_window, image=img_tk)
        label.image = img_tk
        label.pack(padx=10, pady=10)
        
        close_btn = tk.Button(result_window, text="Close", 
                             command=result_window.destroy,
                             font=('Arial', 11, 'bold'),
                             bg='#e74c3c', fg='white', padx=20, pady=5)
        close_btn.pack(pady=10)
    
    def clear_results(self):
        """Clear results panel"""
        self.results_text.delete(1.0, tk.END)
        self.add_result("🗑️ Results cleared.\n\n")
    
    def add_result(self, text):
        """Add text to results panel"""
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
    
    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    
    # Ask for model path
    model_path = 'model/protected_my_model.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        # Try original model
        if os.path.exists('model/my_model.pt'):
            model_path = 'model/my_model.pt'
        else:
            messagebox.showerror("Model Not Found", 
                               "Could not find model file!\n\n"
                               "Expected: model/protected_my_model.pt or model/my_model.pt")
            return
    
    app = CameraDetectionApp(root, model_path)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

