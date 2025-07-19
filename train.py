#!/usr/bin/env python3
"""
Training script for corrosion detection models.
Usage: python train.py
"""

import os
import sys
import argparse

# Add src directory to path
sys.path.append('src')

def main():
    parser = argparse.ArgumentParser(description='Train Corrosion Detection Models')
    parser.add_argument('--data-dir', default='corrosion detect', 
                       help='Directory containing the dataset')
    parser.add_argument('--models-dir', default='models', 
                       help='Directory to save trained models')
    parser.add_argument('--epochs-steel', type=int, default=30,
                       help='Number of epochs for steel classifier')
    parser.add_argument('--epochs-corrosion', type=int, default=50,
                       help='Number of epochs for corrosion detector')
    parser.add_argument('--skip-steel', action='store_true',
                       help='Skip training steel classifier')
    parser.add_argument('--skip-corrosion', action='store_true',
                       help='Skip training corrosion detector')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory not found: {args.data_dir}")
        print("Please ensure your dataset is in the correct location.")
        return 1
    
    images_dir = os.path.join(args.data_dir, "images")
    labels_dir = os.path.join(args.data_dir, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: Images or labels directory not found in {args.data_dir}")
        print("Expected structure:")
        print("  corrosion detect/")
        print("    ├── images/")
        print("    └── labels/")
        return 1
    
    try:
        from train_model import train_corrosion_model
        
        print("Starting training process...")
        print(f"Dataset: {args.data_dir}")
        print(f"Models will be saved to: {args.models_dir}")
        print("-" * 50)
        
        # Train models
        trained_models = train_corrosion_model(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            train_steel_classifier=not args.skip_steel,
            train_corrosion_detector=not args.skip_corrosion
        )
        
        if trained_models:
            print("\n" + "="*60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Models saved in: {args.models_dir}")
            print("\nYou can now make predictions using:")
            print("  python demo.py <image_path>")
            print("\nOr use the Jupyter notebook:")
            print("  jupyter notebook notebooks/corrosion_detection_analysis.ipynb")
            return 0
        else:
            print("❌ Training failed")
            return 1
            
    except ImportError as e:
        print(f"Error importing training module: {e}")
        print("Make sure TensorFlow is installed: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
