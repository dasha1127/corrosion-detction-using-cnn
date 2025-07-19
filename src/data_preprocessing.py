"""
Data preprocessing and loading utilities for corrosion detection.
Handles loading images, parsing YOLO annotations, and data augmentation.
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


class CorrosionDataLoader:
    """
    Data loader for corrosion detection dataset.
    Handles YOLO format annotations and image preprocessing.
    """
    
    def __init__(self, images_dir, labels_dir, img_size=(224, 224)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.steel_labels = []  # For steel/non-steel classification
        
    def load_yolo_annotations(self, label_file):
        """
        Load YOLO format annotations from a text file.
        Returns list of [class_id, x_center, y_center, width, height]
        """
        annotations = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            annotations.append([class_id, x_center, y_center, width, height])
        return annotations
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess an image.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.img_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def create_steel_classification_data(self, non_steel_images_dir=None):
        """
        Create dataset for steel vs non-steel classification.
        If non_steel_images_dir is provided, include those as negative samples.
        """
        steel_images = []
        steel_labels = []
        
        # Load steel images (all images in our corrosion dataset)
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc="Loading steel images"):
            img_path = os.path.join(self.images_dir, img_file)
            image = self.preprocess_image(img_path)
            if image is not None:
                steel_images.append(image)
                steel_labels.append(1)  # Steel = 1
        
        # Load non-steel images if directory provided
        if non_steel_images_dir and os.path.exists(non_steel_images_dir):
            non_steel_files = [f for f in os.listdir(non_steel_images_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm(non_steel_files, desc="Loading non-steel images"):
                img_path = os.path.join(non_steel_images_dir, img_file)
                image = self.preprocess_image(img_path)
                if image is not None:
                    steel_images.append(image)
                    steel_labels.append(0)  # Non-steel = 0
        
        return np.array(steel_images), np.array(steel_labels)
    
    def create_corrosion_detection_data(self):
        """
        Create dataset for corrosion detection and localization.
        """
        images = []
        classifications = []  # 0: no corrosion, 1: corrosion present
        bboxes = []  # Bounding box coordinates
        
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc="Loading corrosion data"):
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            
            # Load image
            img_path = os.path.join(self.images_dir, img_file)
            image = self.preprocess_image(img_path)
            if image is None:
                continue
            
            # Load annotations
            annotations = self.load_yolo_annotations(label_path)
            
            if len(annotations) > 0:
                # Corrosion present
                classifications.append(1)
                # Use first annotation for simplicity (multi-object detection can be added later)
                bbox = annotations[0][1:]  # x_center, y_center, width, height
                bboxes.append(bbox)
            else:
                # No corrosion
                classifications.append(0)
                bboxes.append([0.0, 0.0, 0.0, 0.0])  # Dummy bbox
            
            images.append(image)
        
        return np.array(images), np.array(classifications), np.array(bboxes)
    
    def augment_data(self, images, labels, bboxes=None, augmentation_factor=2):
        """
        Apply data augmentation to increase dataset size.
        """
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        augmented_images = []
        augmented_labels = []
        augmented_bboxes = []
        
        # Original data
        augmented_images.extend(images)
        augmented_labels.extend(labels)
        if bboxes is not None:
            augmented_bboxes.extend(bboxes)
        
        # Generate augmented data
        for i in range(len(images)):
            img = images[i].reshape(1, *images[i].shape)
            
            # Generate augmented versions
            aug_iter = datagen.flow(img, batch_size=1)
            
            for j in range(augmentation_factor):
                aug_img = next(aug_iter)[0]
                augmented_images.append(aug_img)
                augmented_labels.append(labels[i])
                if bboxes is not None:
                    augmented_bboxes.append(bboxes[i])
        
        if bboxes is not None:
            return np.array(augmented_images), np.array(augmented_labels), np.array(augmented_bboxes)
        else:
            return np.array(augmented_images), np.array(augmented_labels)
    
    def prepare_steel_classification_data(self, test_size=0.2, augment=True):
        """
        Prepare data for steel classification model.
        """
        # Create steel classification dataset
        images, labels = self.create_steel_classification_data()
        
        # Convert labels to categorical
        labels_categorical = to_categorical(labels, num_classes=2)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels_categorical, test_size=test_size, 
            stratify=labels, random_state=42
        )
        
        # Apply augmentation to training data
        if augment:
            X_train, y_train = self.augment_data(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_corrosion_detection_data(self, test_size=0.2, augment=True):
        """
        Prepare data for corrosion detection model.
        """
        # Create corrosion detection dataset
        images, classifications, bboxes = self.create_corrosion_detection_data()
        
        # Convert classifications to categorical
        classifications_categorical = to_categorical(classifications, num_classes=2)
        
        # Split data
        X_train, X_test, y_class_train, y_class_test, y_bbox_train, y_bbox_test = train_test_split(
            images, classifications_categorical, bboxes, 
            test_size=test_size, stratify=classifications, random_state=42
        )
        
        # Apply augmentation to training data
        if augment:
            X_train, y_class_train, y_bbox_train = self.augment_data(
                X_train, y_class_train, y_bbox_train
            )
        
        return (X_train, X_test, 
                {'classification': y_class_train, 'regression': y_bbox_train},
                {'classification': y_class_test, 'regression': y_bbox_test})


def create_synthetic_non_steel_data(output_dir, num_images=100, img_size=(224, 224)):
    """
    Create synthetic non-steel images for training the steel classifier.
    This generates random textures, patterns, and colors that don't look like steel.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(num_images), desc="Creating synthetic non-steel images"):
        # Create random image
        if random.choice([True, False]):
            # Generate random noise
            image = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
        else:
            # Generate gradient or pattern
            image = np.zeros((*img_size, 3), dtype=np.uint8)
            
            # Random colors (avoid metallic colors)
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255],  # RGB
                [255, 255, 0], [255, 0, 255], [0, 255, 255],  # CMY
                [139, 69, 19], [34, 139, 34], [255, 165, 0]  # Brown, Green, Orange
            ]
            
            color = random.choice(colors)
            image[:, :] = color
            
            # Add some texture
            noise = np.random.normal(0, 30, img_size + (3,))
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Save image
        filename = f"non_steel_{i:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def visualize_data_samples(images, labels, bboxes=None, num_samples=8):
    """
    Visualize some data samples with their labels and bounding boxes.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx])
        
        title = f"Label: {np.argmax(labels[idx])}"
        if bboxes is not None and np.sum(bboxes[idx]) > 0:
            # Draw bounding box
            bbox = bboxes[idx]
            x_center, y_center, width, height = bbox
            
            # Convert from normalized to pixel coordinates
            img_h, img_w = images[idx].shape[:2]
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h
            
            # Calculate corner coordinates
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # Draw rectangle
            from matplotlib.patches import Rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
            
            title += f"\nBBox: ({x1},{y1},{x2},{y2})"
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test data loading
    images_dir = "../corrosion detect/images"
    labels_dir = "../corrosion detect/labels"
    
    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        loader = CorrosionDataLoader(images_dir, labels_dir)
        
        print("Testing steel classification data preparation...")
        X_train, X_test, y_train, y_test = loader.prepare_steel_classification_data()
        print(f"Steel classification data: Train {X_train.shape}, Test {X_test.shape}")
        
        print("\nTesting corrosion detection data preparation...")
        train_data, test_data, train_labels, test_labels = loader.prepare_corrosion_detection_data()
        print(f"Corrosion detection data: Train {train_data.shape}, Test {test_data.shape}")
        
        # Visualize some samples
        print("\nVisualizing samples...")
        visualize_data_samples(X_train[:8], y_train[:8])
    else:
        print("Data directories not found. Please check the paths.")
