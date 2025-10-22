#!/usr/bin/env python3
"""
Secure Model Loader
Automatically handles protected models
"""

import os
import atexit
from protect_models import ModelProtector

class SecureModelLoader:
    """Load protected YOLO models securely"""
    
    def __init__(self):
        self.protector = ModelProtector()
        self.temp_files = []
    
    def load_model(self, model_path):
        """Load a protected or regular model"""
        from ultralytics import YOLO
        
        # Check if it's a protected model
        if model_path.startswith('protected_') or os.path.exists(model_path + '.lock'):
            print(f"🔓 Loading protected model: {model_path}")
            
            try:
                # Decrypt to temp file
                temp_path = self.protector.decrypt_model(model_path)
                self.temp_files.append(temp_path)
                
                # Load from temp file
                model = YOLO(temp_path)
                
                print(f"✅ Model loaded successfully")
                return model
                
            except PermissionError as e:
                print(f"\n{'='*70}")
                print("❌ MODEL AUTHORIZATION FAILED")
                print('='*70)
                print(str(e))
                print(f"{'='*70}")
                raise
            except Exception as e:
                print(f"❌ Failed to load protected model: {e}")
                raise
        else:
            # Regular model
            return YOLO(model_path)
    
    def cleanup(self):
        """Clean up temporary decrypted files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
    
    def __del__(self):
        self.cleanup()

# Register cleanup
_loader = None

def get_loader():
    global _loader
    if _loader is None:
        _loader = SecureModelLoader()
        atexit.register(_loader.cleanup)
    return _loader

def load_protected_model(model_path):
    """Convenience function to load protected models"""
    loader = get_loader()
    return loader.load_model(model_path)
