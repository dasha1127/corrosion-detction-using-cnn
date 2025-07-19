"""
Model architecture for corrosion detection using CNN.
This module contains the neural network architectures for both steel classification
and corrosion detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


class SteelClassifier:
    """
    CNN model to classify if an image contains steel surfaces.
    This is the first stage of our pipeline.
    """
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """Build the steel classification CNN model."""
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer (binary classification: steel vs non-steel)
            layers.Dense(2, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
    def get_model(self):
        """Return the compiled model."""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model


class CorrosionDetector:
    """
    CNN model for corrosion detection and localization.
    This is the second stage of our pipeline, used only when steel is detected.
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes  # background, corrosion
        self.model = None
        
    def build_model(self):
        """Build the corrosion detection CNN model."""
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Backbone CNN feature extractor
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Feature maps for detection
        feature_maps = x
        
        # Classification head (corrosion present/absent)
        classification = layers.GlobalAveragePooling2D()(feature_maps)
        classification = layers.Dense(256, activation='relu')(classification)
        classification = layers.BatchNormalization()(classification)
        classification = layers.Dropout(0.5)(classification)
        classification = layers.Dense(128, activation='relu')(classification)
        classification = layers.BatchNormalization()(classification)
        classification = layers.Dropout(0.5)(classification)
        classification_output = layers.Dense(self.num_classes, activation='softmax', name='classification')(classification)
        
        # Regression head (bounding box coordinates)
        regression = layers.GlobalAveragePooling2D()(feature_maps)
        regression = layers.Dense(256, activation='relu')(regression)
        regression = layers.BatchNormalization()(regression)
        regression = layers.Dropout(0.5)(regression)
        regression = layers.Dense(128, activation='relu')(regression)
        regression = layers.BatchNormalization()(regression)
        regression = layers.Dropout(0.5)(regression)
        # 4 coordinates: x_center, y_center, width, height
        regression_output = layers.Dense(4, activation='sigmoid', name='regression')(regression)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=[classification_output, regression_output])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with multi-task loss."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        losses = {
            'classification': 'categorical_crossentropy',
            'regression': 'mse'
        }
        
        loss_weights = {
            'classification': 1.0,
            'regression': 1.0
        }
        
        metrics = {
            'classification': ['accuracy'],
            'regression': ['mae']
        }
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
    def get_model(self):
        """Return the compiled model."""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model


class TransferLearningCorrosionDetector:
    """
    Transfer learning based corrosion detector using pre-trained models.
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self, base_model_name='EfficientNetB0'):
        """Build model using transfer learning."""
        
        # Load pre-trained base model
        if base_model_name == 'EfficientNetB0':
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'ResNet50':
            base_model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError("Unsupported base model")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom head
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification branch
        classification = layers.Dense(256, activation='relu')(x)
        classification = layers.BatchNormalization()(classification)
        classification = layers.Dropout(0.5)(classification)
        classification_output = layers.Dense(self.num_classes, activation='softmax', name='classification')(classification)
        
        # Regression branch
        regression = layers.Dense(256, activation='relu')(x)
        regression = layers.BatchNormalization()(regression)
        regression = layers.Dropout(0.5)(regression)
        regression_output = layers.Dense(4, activation='sigmoid', name='regression')(regression)
        
        model = keras.Model(inputs, [classification_output, regression_output])
        
        self.model = model
        self.base_model = base_model
        return model
    
    def unfreeze_base_model(self):
        """Unfreeze base model for fine-tuning."""
        self.base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(self.base_model.layers) // 2
        
        # Freeze all the layers before fine_tune_at
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
    
    def compile_model(self, learning_rate=0.0001):
        """Compile the model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        losses = {
            'classification': 'categorical_crossentropy',
            'regression': 'mse'
        }
        
        loss_weights = {
            'classification': 1.0,
            'regression': 1.0
        }
        
        metrics = {
            'classification': ['accuracy'],
            'regression': ['mae']
        }
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
    
    def get_model(self):
        """Return the compiled model."""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model


def create_ensemble_model(input_shape=(224, 224, 3)):
    """
    Create an ensemble model combining multiple architectures.
    """
    
    # Create multiple models
    model1 = CorrosionDetector(input_shape).get_model()
    model2 = TransferLearningCorrosionDetector(input_shape).get_model()
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Get predictions from both models
    pred1_class, pred1_reg = model1(inputs)
    pred2_class, pred2_reg = model2(inputs)
    
    # Average predictions
    avg_classification = layers.Average()([pred1_class, pred2_class])
    avg_regression = layers.Average()([pred1_reg, pred2_reg])
    
    # Create ensemble model
    ensemble_model = models.Model(inputs=inputs, outputs=[avg_classification, avg_regression])
    
    return ensemble_model


if __name__ == "__main__":
    # Test model creation
    print("Testing Steel Classifier...")
    steel_classifier = SteelClassifier()
    steel_model = steel_classifier.get_model()
    print(f"Steel Classifier created with {steel_model.count_params()} parameters")
    
    print("\nTesting Corrosion Detector...")
    corrosion_detector = CorrosionDetector()
    corrosion_model = corrosion_detector.get_model()
    print(f"Corrosion Detector created with {corrosion_model.count_params()} parameters")
    
    print("\nTesting Transfer Learning Model...")
    transfer_model = TransferLearningCorrosionDetector()
    transfer_model_instance = transfer_model.get_model()
    print(f"Transfer Learning Model created with {transfer_model_instance.count_params()} parameters")
