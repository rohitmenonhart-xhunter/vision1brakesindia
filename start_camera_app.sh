#!/bin/bash
# Launcher script for Camera Detection App

echo "=================================="
echo "🎥 Starting Camera Detection App"
echo "=================================="
echo ""

# Activate conda environment and run app
conda activate yolo_env
python camera_detection_app.py

