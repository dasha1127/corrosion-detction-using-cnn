"""
Prediction module for corrosion detection.
Provides functions to make predictions on new images using trained models.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not installed. Please install it to use this module.")


class CorrosionPredictor:
    """
    Main predictor class for corrosion detection.
    Combines steel classification and corrosion detection models.
    """
    
    def __init__(self, models_dir="models", img_size=(224, 224)):
        self.models_dir = models_dir
        self.img_size = img_size
        self.steel_classifier = None
        self.corrosion_detector = None
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """
        Load trained models from disk.
        """
        if not HAS_TENSORFLOW:
            print("TensorFlow is required for predictions. Please install it.")
            return
        
        # Load steel classifier
        steel_classifier_path = os.path.join(self.models_dir, "steel_classifier_best.h5")
        if not os.path.exists(steel_classifier_path):
            steel_classifier_path = os.path.join(self.models_dir, "steel_classifier_final.h5")
        
        if os.path.exists(steel_classifier_path):
            try:
                self.steel_classifier = tf.keras.models.load_model(steel_classifier_path)
                print(f"Steel classifier loaded from {steel_classifier_path}")
            except Exception as e:
                print(f"Error loading steel classifier: {e}")
        else:
            print("Steel classifier model not found. Please train the model first.")
        
        # Load corrosion detector
        corrosion_detector_path = os.path.join(self.models_dir, "corrosion_detector_transfer_best.h5")
        if not os.path.exists(corrosion_detector_path):
            corrosion_detector_path = os.path.join(self.models_dir, "corrosion_detector_transfer_final.h5")
        if not os.path.exists(corrosion_detector_path):
            corrosion_detector_path = os.path.join(self.models_dir, "corrosion_detector_custom_best.h5")
        if not os.path.exists(corrosion_detector_path):
            corrosion_detector_path = os.path.join(self.models_dir, "corrosion_detector_custom_final.h5")
        
        if os.path.exists(corrosion_detector_path):
            try:
                self.corrosion_detector = tf.keras.models.load_model(corrosion_detector_path)
                print(f"Corrosion detector loaded from {corrosion_detector_path}")
            except Exception as e:
                print(f"Error loading corrosion detector: {e}")
        else:
            print("Corrosion detector model not found. Please train the model first.")
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path
        
        # Store original size for later use
        original_shape = image.shape[:2]
        
        # Resize image
        image_resized = cv2.resize(image, self.img_size)
        
        # Normalize pixel values to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, image_resized, original_shape
    
    def predict_steel(self, image_path, threshold=0.5):
        """
        Predict if an image contains steel surfaces.
        
        Args:
            image_path: Path to the image or numpy array
            threshold: Confidence threshold for steel classification
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if not HAS_TENSORFLOW or self.steel_classifier is None:
            return {"error": "Steel classifier not available"}
        
        try:
            # Preprocess image
            image_batch, _, _ = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.steel_classifier.predict(image_batch, verbose=0)
            
            # Extract probabilities
            non_steel_prob = prediction[0][0]
            steel_prob = prediction[0][1]
            
            # Determine prediction
            is_steel = steel_prob > threshold
            confidence = steel_prob if is_steel else non_steel_prob
            
            return {
                "is_steel": is_steel,
                "confidence": float(confidence),
                "steel_probability": float(steel_prob),
                "non_steel_probability": float(non_steel_prob)
            }
            
        except Exception as e:
            return {"error": f"Error in steel prediction: {str(e)}"}
    
    def predict_corrosion(self, image_path, threshold=0.5):
        """
        Predict corrosion in an image (assumes image contains steel).
        
        Args:
            image_path: Path to the image or numpy array
            threshold: Confidence threshold for corrosion detection
            
        Returns:
            dict: Prediction results with bounding box and confidence
        """
        if not HAS_TENSORFLOW or self.corrosion_detector is None:
            return {"error": "Corrosion detector not available"}
        
        try:
            # Preprocess image
            image_batch, image_resized, original_shape = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.corrosion_detector.predict(image_batch, verbose=0)
            
            if isinstance(prediction, list):
                classification_pred, bbox_pred = prediction
            else:
                classification_pred = prediction
                bbox_pred = None
            
            # Extract classification probabilities
            no_corrosion_prob = classification_pred[0][0]
            corrosion_prob = classification_pred[0][1]
            
            # Determine if corrosion is present
            has_corrosion = corrosion_prob > threshold
            confidence = corrosion_prob if has_corrosion else no_corrosion_prob
            
            result = {
                "has_corrosion": has_corrosion,
                "confidence": float(confidence),
                "corrosion_probability": float(corrosion_prob),
                "no_corrosion_probability": float(no_corrosion_prob)
            }
            
            # Add bounding box if available and corrosion detected
            if bbox_pred is not None and has_corrosion:
                bbox = bbox_pred[0]
                x_center, y_center, width, height = bbox
                
                # Convert normalized coordinates to pixel coordinates
                original_h, original_w = original_shape
                x_center_px = x_center * original_w
                y_center_px = y_center * original_h
                width_px = width * original_w
                height_px = height * original_h
                
                # Calculate corner coordinates
                x1 = max(0, int(x_center_px - width_px/2))
                y1 = max(0, int(y_center_px - height_px/2))
                x2 = min(original_w, int(x_center_px + width_px/2))
                y2 = min(original_h, int(y_center_px + height_px/2))
                
                result["bounding_box"] = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "x_center": float(x_center_px),
                    "y_center": float(y_center_px),
                    "width": float(width_px),
                    "height": float(height_px)
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Error in corrosion prediction: {str(e)}"}
    
    def predict_full_pipeline(self, image_path, steel_threshold=0.7, corrosion_threshold=0.5):
        """
        Complete prediction pipeline: first check for steel, then for corrosion.
        
        Args:
            image_path: Path to the image
            steel_threshold: Confidence threshold for steel classification
            corrosion_threshold: Confidence threshold for corrosion detection
            
        Returns:
            dict: Complete prediction results
        """
        # Step 1: Check if image contains steel
        steel_result = self.predict_steel(image_path, steel_threshold)
        
        if "error" in steel_result:
            return steel_result
        
        # If not steel, return "Not Applicable"
        if not steel_result["is_steel"]:
            return {
                "status": "Not Applicable",
                "message": "Image does not contain steel surfaces",
                "steel_classification": steel_result
            }
        
        # Step 2: Check for corrosion
        corrosion_result = self.predict_corrosion(image_path, corrosion_threshold)
        
        if "error" in corrosion_result:
            return corrosion_result
        
        # Combine results
        result = {
            "status": "Steel Detected",
            "steel_classification": steel_result,
            "corrosion_detection": corrosion_result
        }
        
        if corrosion_result["has_corrosion"]:
            result["final_result"] = "Corrosion Detected"
            result["severity"] = self._assess_corrosion_severity(corrosion_result)
        else:
            result["final_result"] = "No Corrosion Detected"
        
        return result
    
    def _assess_corrosion_severity(self, corrosion_result):
        """
        Assess corrosion severity based on confidence and bounding box size.
        """
        confidence = corrosion_result["confidence"]
        
        if "bounding_box" in corrosion_result:
            bbox = corrosion_result["bounding_box"]
            area = bbox["width"] * bbox["height"]
            # Normalize area (assuming 224x224 input)
            normalized_area = area / (224 * 224)
        else:
            normalized_area = 0
        
        # Simple severity assessment
        if confidence > 0.9 and normalized_area > 0.1:
            return "High"
        elif confidence > 0.7 and normalized_area > 0.05:
            return "Medium"
        else:
            return "Low"
    
    def visualize_prediction(self, image_path, prediction_result, save_path=None):
        """
        Visualize prediction results on the image.
        """
        # Load original image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image)
        
        # Add title based on prediction
        if prediction_result.get("status") == "Not Applicable":
            title = "Not Applicable - No Steel Detected"
            color = 'orange'
        elif prediction_result.get("final_result") == "Corrosion Detected":
            title = f"Corrosion Detected - Severity: {prediction_result.get('severity', 'Unknown')}"
            color = 'red'
            
            # Draw bounding box if available
            if "corrosion_detection" in prediction_result and "bounding_box" in prediction_result["corrosion_detection"]:
                bbox = prediction_result["corrosion_detection"]["bounding_box"]
                rect = patches.Rectangle(
                    (bbox["x1"], bbox["y1"]),
                    bbox["x2"] - bbox["x1"],
                    bbox["y2"] - bbox["y1"],
                    linewidth=3,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add confidence text
                confidence = prediction_result["corrosion_detection"]["confidence"]
                ax.text(bbox["x1"], bbox["y1"]-10, f'Confidence: {confidence:.2f}', 
                       color='red', fontsize=12, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            title = "No Corrosion Detected"
            color = 'green'
        
        ax.set_title(title, fontsize=16, weight='bold', color=color)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return fig


def predict_corrosion(image_path, models_dir="models", visualize=True, save_visualization=None):
    """
    Convenience function for making predictions on a single image.
    
    Args:
        image_path: Path to the image
        models_dir: Directory containing trained models
        visualize: Whether to show visualization
        save_visualization: Path to save visualization (optional)
    
    Returns:
        dict: Prediction results
    """
    # Create predictor
    predictor = CorrosionPredictor(models_dir)
    
    # Make prediction
    result = predictor.predict_full_pipeline(image_path)
    
    # Print results
    print("\n" + "="*50)
    print("CORROSION DETECTION RESULTS")
    print("="*50)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return result
    
    print(f"Status: {result['status']}")
    
    if result['status'] == "Not Applicable":
        print(f"Reason: {result['message']}")
        steel_conf = result['steel_classification']['confidence']
        print(f"Steel Detection Confidence: {steel_conf:.3f}")
    else:
        print(f"Final Result: {result['final_result']}")
        
        # Steel classification details
        steel = result['steel_classification']
        print(f"Steel Detection Confidence: {steel['confidence']:.3f}")
        
        # Corrosion detection details
        corrosion = result['corrosion_detection']
        print(f"Corrosion Detection Confidence: {corrosion['confidence']:.3f}")
        
        if corrosion['has_corrosion']:
            print(f"Severity: {result.get('severity', 'Unknown')}")
            
            if 'bounding_box' in corrosion:
                bbox = corrosion['bounding_box']
                print(f"Corrosion Location: ({bbox['x1']}, {bbox['y1']}) to ({bbox['x2']}, {bbox['y2']})")
    
    # Visualize if requested
    if visualize:
        predictor.visualize_prediction(image_path, result, save_visualization)
    
    return result


def batch_predict(image_directory, models_dir="models", output_file="predictions.json"):
    """
    Make predictions on a batch of images.
    
    Args:
        image_directory: Directory containing images
        models_dir: Directory containing trained models
        output_file: File to save results
    
    Returns:
        dict: Results for all images
    """
    predictor = CorrosionPredictor(models_dir)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(image_directory) 
                  if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    results = {}
    
    print(f"Processing {len(image_files)} images...")
    
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file}")
        
        image_path = os.path.join(image_directory, image_file)
        result = predictor.predict_full_pipeline(image_path)
        results[image_file] = result
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    total = len(results)
    not_applicable = sum(1 for r in results.values() if r.get('status') == 'Not Applicable')
    corrosion_detected = sum(1 for r in results.values() if r.get('final_result') == 'Corrosion Detected')
    no_corrosion = total - not_applicable - corrosion_detected
    
    print("\n" + "="*40)
    print("BATCH PROCESSING SUMMARY")
    print("="*40)
    print(f"Total Images: {total}")
    print(f"Not Applicable (No Steel): {not_applicable}")
    print(f"Corrosion Detected: {corrosion_detected}")
    print(f"No Corrosion: {no_corrosion}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            result = predict_corrosion(image_path)
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py ../corrosion detect/images/image1.jpeg")
