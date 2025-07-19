#!/usr/bin/env python3
"""
Demo script for corrosion detection.
Usage: python demo.py <image_path>
"""

import os
import sys
import argparse
import numpy as np
import cv2

# Add src directory to path
sys.path.append('src')

def main():
    parser = argparse.ArgumentParser(description='Corrosion Detection Demo')
    parser.add_argument('image_path', help='Path to the image to analyze')
    parser.add_argument('--models-dir', default='models', help='Directory containing trained models')
    parser.add_argument('--save-viz', help='Path to save visualization (optional)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return 1
    
    # Check if models exist
    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory not found: {args.models_dir}")
        print("Please train the models first using: python src/train_model.py")
        return 1
    
    try:
        from predict import predict_corrosion
        
        # Make prediction
        print(f"Analyzing image: {args.image_path}")
        print("-" * 50)
        
        result = predict_corrosion(
            args.image_path, 
            models_dir=args.models_dir,
            visualize=not args.no_viz,
            save_visualization=args.save_viz
        )
        
        return 0 if 'error' not in result else 1
        
    except ImportError as e:
        print(f"Error importing prediction module: {e}")
        print("Make sure TensorFlow is installed: pip install tensorflow")
        return 1
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
