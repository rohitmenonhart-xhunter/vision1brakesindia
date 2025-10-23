#!/usr/bin/env python3
"""
Model Protection System
Encrypts YOLO models and binds them to specific hardware
Prevents unauthorized sharing and usage
"""

import os
import uuid
import hashlib
import json
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64


class ModelProtector:
    """Protect YOLO models with encryption and hardware binding"""
    
    def __init__(self):
        self.config_file = '.model_protection.json'
        
    def get_hardware_id(self):
        """Get unique hardware identifier"""
        # Combine multiple hardware identifiers for security
        mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff)
                       for elements in range(0, 2*6, 2)][::-1])
        
        # Get system UUID (more secure)
        try:
            import platform
            system_info = platform.uname()
            hw_string = f"{mac}_{system_info.node}_{system_info.machine}"
        except:
            hw_string = mac
        
        # Create hash of hardware info
        hw_hash = hashlib.sha256(hw_string.encode()).hexdigest()
        return hw_hash[:32]
    
    def generate_key(self, hardware_id, password="omni_secure_2025"):
        """Generate encryption key from hardware ID and password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=hardware_id.encode(),
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_model(self, model_path, output_path=None):
        """Encrypt a model file"""
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False
        
        # Get hardware ID
        hw_id = self.get_hardware_id()
        
        # Generate encryption key
        key = self.generate_key(hw_id)
        fernet = Fernet(key)
        
        # Read model file
        print(f"🔒 Encrypting: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Encrypt
        encrypted_data = fernet.encrypt(model_data)
        
        # Create output path
        if output_path is None:
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path)
            output_path = os.path.join(model_dir, f"protected_{model_name}")
        
        # Save encrypted model
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        # Save protection info
        protection_info = {
            'original_model': model_path,
            'protected_model': output_path,
            'hardware_id': hw_id,
            'protected': True
        }
        
        info_path = output_path + '.lock'
        with open(info_path, 'w') as f:
            json.dump(protection_info, f, indent=2)
        
        print(f"✅ Encrypted model saved: {output_path}")
        print(f"🔐 Lock file saved: {info_path}")
        print(f"🔑 Hardware ID: {hw_id[:16]}...")
        
        return output_path
    
    def decrypt_model(self, encrypted_path, temp_path=None):
        """Decrypt model for use (only works on authorized hardware)"""
        if not os.path.exists(encrypted_path):
            raise FileNotFoundError(f"Protected model not found: {encrypted_path}")
        
        # Load protection info
        info_path = encrypted_path + '.lock'
        if not os.path.exists(info_path):
            raise FileNotFoundError("Protection info missing! Model may be corrupted.")
        
        with open(info_path, 'r') as f:
            protection_info = json.load(f)
        
        # Verify hardware ID
        current_hw_id = self.get_hardware_id()
        authorized_hw_id = protection_info['hardware_id']
        
        if current_hw_id != authorized_hw_id:
            raise PermissionError(
                "❌ UNAUTHORIZED HARDWARE!\n"
                "This model is locked to different hardware.\n"
                "It cannot be used on this system."
            )
        
        # Generate decryption key
        key = self.generate_key(current_hw_id)
        fernet = Fernet(key)
        
        # Read encrypted data
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt
        try:
            decrypted_data = fernet.decrypt(encrypted_data)
        except Exception as e:
            raise PermissionError("Decryption failed! Model may be tampered with.")
        
        # Save to temporary location
        if temp_path is None:
            temp_path = encrypted_path.replace('protected_', 'temp_')
        
        with open(temp_path, 'wb') as f:
            f.write(decrypted_data)
        
        return temp_path
    
    def protect_all_models(self, base_dir='.'):
        """Find and protect all .pt model files"""
        print("=" * 70)
        print("🔒 MODEL PROTECTION SYSTEM")
        print("=" * 70)
        
        # Find all .pt files
        model_files = list(Path(base_dir).rglob('*.pt'))
        
        # Exclude already protected files
        model_files = [m for m in model_files if not m.name.startswith('protected_')
                      and not m.name.startswith('temp_')]
        
        if not model_files:
            print("❌ No unprotected .pt files found")
            return
        
        print(f"\n📦 Found {len(model_files)} model(s):")
        for i, model in enumerate(model_files, 1):
            print(f"  {i}. {model}")
        
        # Get hardware ID
        hw_id = self.get_hardware_id()
        print(f"\n🔑 This system's Hardware ID: {hw_id[:16]}...")
        
        # Confirm
        print("\n⚠️  WARNING:")
        print("   Protected models will ONLY work on THIS computer!")
        print("   They cannot be shared or used on other systems.")
        
        confirm = input("\n🔒 Protect all models? (yes/no): ").strip().lower()
        
        if confirm not in ['yes', 'y']:
            print("❌ Protection cancelled")
            return
        
        # Protect each model
        protected_count = 0
        for model_path in model_files:
            try:
                protected_path = self.encrypt_model(str(model_path))
                if protected_path:
                    protected_count += 1
            except Exception as e:
                print(f"❌ Failed to protect {model_path}: {e}")
        
        print(f"\n✅ Protected {protected_count}/{len(model_files)} models")
        
        # Save master config
        config = {
            'hardware_id': hw_id,
            'protected_models': protected_count,
            'system_locked': True
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Configuration saved: {self.config_file}")
        print("\n" + "=" * 70)
        print("🎯 NEXT STEPS:")
        print("=" * 70)
        print("1. Original models are still in place")
        print("2. Protected models created with 'protected_' prefix")
        print("3. Update your scripts to use protected models")
        print("4. Delete original .pt files after testing")
        print("5. Distribute")
        print("=" * 70)


def create_secure_loader():
    """Create a secure model loader wrapper"""
    
    loader_code = '''#!/usr/bin/env python3
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
                print(f"\\n{'='*70}")
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
'''
    
    with open('secure_loader.py', 'w', encoding='utf-8') as f:
        f.write(loader_code)
    
    print("✅ Created secure_loader.py")


def main():
    """Main protection function"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    MODEL PROTECTION SYSTEM                           ║
║                  Hardware-Locked Model Encryption                    ║
╚══════════════════════════════════════════════════════════════════════╝

This system will:
✅ Encrypt all your YOLO model files
✅ Bind them to THIS computer's hardware
✅ Prevent unauthorized sharing
✅ Block usage on other systems

Protected models will:
❌ NOT work if copied to another computer
❌ NOT work if hardware changes
❌ NOT be readable without the correct system
✅ ONLY work on THIS authorized computer

""")
    
    # Check if cryptography is installed
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        print("❌ Required package 'cryptography' not installed")
        print("\nInstall it with:")
        print("   conda activate yolo_env")
        print("   pip install cryptography")
        return
    
    # Create protector
    protector = ModelProtector()
    
    # Protect all models
    protector.protect_all_models('model')
    
    # Create secure loader
    create_secure_loader()
    
    print("\n✅ Model protection complete!")
    print("\n📖 See SECURITY_GUIDE.md for usage instructions")


if __name__ == '__main__':
    main()

