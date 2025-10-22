#!/usr/bin/env python3
"""
Client Setup Script - First Run Protection
Automatically protects models on client's computer on first run
"""

import os
import sys
from pathlib import Path
from protect_models import ModelProtector

def check_if_protected():
    """Check if models are already protected"""
    model_files = list(Path('model').rglob('*.pt'))
    protected_files = [f for f in model_files if f.name.startswith('protected_')]
    original_files = [f for f in model_files if not f.name.startswith('protected_') 
                     and not f.name.startswith('temp_')]
    
    return len(protected_files) > 0, len(original_files) > 0

def first_time_setup():
    """Run first-time setup for client"""
    print("=" * 70)
    print("🔧 FIRST TIME SETUP - MODEL PROTECTION")
    print("=" * 70)
    print("\nThis is the first time running on this computer.")
    print("Models will be locked to THIS computer's hardware.\n")
    
    # Check for models
    has_protected, has_original = check_if_protected()
    
    if not has_original:
        print("❌ No model files found to protect!")
        print("   Please ensure .pt files are in the model/ directory")
        return False
    
    print(f"📦 Found unprotected models")
    print("\n⚠️  IMPORTANT:")
    print("   After protection, these models will ONLY work on THIS computer.")
    print("   They cannot be shared or used on other systems.\n")
    
    # Auto-confirm for client
    print("🔒 Protecting models for this computer...")
    
    try:
        protector = ModelProtector()
        
        # Find all .pt files
        model_files = list(Path('model').rglob('*.pt'))
        model_files = [m for m in model_files if not m.name.startswith('protected_')
                      and not m.name.startswith('temp_')]
        
        protected_count = 0
        for model_path in model_files:
            try:
                print(f"\n🔒 Protecting: {model_path}")
                protected_path = protector.encrypt_model(str(model_path))
                if protected_path:
                    protected_count += 1
                    # Delete original after successful protection
                    try:
                        os.remove(str(model_path))
                        print(f"   ✅ Protected and secured")
                    except:
                        print(f"   ⚠️  Protected but could not delete original")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        if protected_count > 0:
            print(f"\n✅ Successfully protected {protected_count} model(s)")
            print("🔐 Models are now locked to this computer")
            
            # Create marker file
            with open('.models_protected', 'w') as f:
                f.write('Models protected successfully')
            
            return True
        else:
            print("\n❌ No models were protected")
            return False
            
    except Exception as e:
        print(f"\n❌ Protection failed: {e}")
        return False

def main():
    """Main setup check"""
    
    # Check if already set up
    if os.path.exists('.models_protected'):
        # Already protected
        return True
    
    # First time - run setup
    try:
        success = first_time_setup()
        if success:
            print("\n" + "=" * 70)
            print("✅ SETUP COMPLETE")
            print("=" * 70)
            print("\nYou can now use the application normally.")
            print("Models are protected and ready to use.")
            print("=" * 70)
        else:
            print("\n⚠️  Setup incomplete. Please contact support.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

