"""
Training script for corrosion detection models.
Handles training of both steel classification and corrosion detection models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import json

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not installed. Please install it to use this module.")

from data_preprocessing import CorrosionDataLoader, create_synthetic_non_steel_data
from model_architecture import SteelClassifier, CorrosionDetector, TransferLearningCorrosionDetector


class ModelTrainer:
    """
    Handles training of corrosion detection models.
    """
    
    def __init__(self, data_dir, models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = CorrosionDataLoader(self.images_dir, self.labels_dir)
        
        # Training history
        self.history = {}
    
    def create_callbacks(self, model_name, patience=10):
        """Create training callbacks."""
        if not HAS_TENSORFLOW:
            return []
            
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, f"{model_name}_best.h5"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_steel_classifier(self, epochs=50, batch_size=32, create_synthetic=True):
        """
        Train the steel classification model.
        """
        if not HAS_TENSORFLOW:
            print("TensorFlow is required for training. Please install it.")
            return None
            
        print("Training Steel Classifier...")
        
        # Create synthetic non-steel data if requested
        if create_synthetic:
            synthetic_dir = os.path.join(self.data_dir, "non_steel_synthetic")
            create_synthetic_non_steel_data(synthetic_dir, num_images=200)
            
            # Update data loader to include synthetic data
            self.data_loader = CorrosionDataLoader(self.images_dir, self.labels_dir)
        
        # Prepare data
        print("Preparing steel classification data...")
        X_train, X_test, y_train, y_test = self.data_loader.prepare_steel_classification_data()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Create and compile model
        steel_classifier = SteelClassifier()
        model = steel_classifier.get_model()
        
        print("Model summary:")
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks("steel_classifier")
        
        # Train model
        print("Starting training...")
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.history['steel_classifier'] = history.history
        
        # Save final model
        model.save(os.path.join(self.models_dir, "steel_classifier_final.h5"))
        
        # Evaluate model
        self.evaluate_steel_classifier(model, X_test, y_test)
        
        return model, history
    
    def train_corrosion_detector(self, epochs=100, batch_size=16, use_transfer_learning=True):
        """
        Train the corrosion detection model.
        """
        if not HAS_TENSORFLOW:
            print("TensorFlow is required for training. Please install it.")
            return None
            
        print("Training Corrosion Detector...")
        
        # Prepare data
        print("Preparing corrosion detection data...")
        train_data, test_data, train_labels, test_labels = self.data_loader.prepare_corrosion_detection_data()
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Create and compile model
        if use_transfer_learning:
            detector = TransferLearningCorrosionDetector()
            model_name = "corrosion_detector_transfer"
        else:
            detector = CorrosionDetector()
            model_name = "corrosion_detector_custom"
        
        model = detector.get_model()
        
        print("Model summary:")
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Train model
        print("Starting training...")
        history = model.fit(
            train_data, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_data, test_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tune if using transfer learning
        if use_transfer_learning and hasattr(detector, 'unfreeze_base_model'):
            print("Fine-tuning with unfrozen base model...")
            detector.unfreeze_base_model()
            detector.compile_model(learning_rate=0.0001/10)  # Lower learning rate for fine-tuning
            
            # Continue training
            history_finetune = model.fit(
                train_data, train_labels,
                batch_size=batch_size,
                epochs=epochs//2,  # Fewer epochs for fine-tuning
                validation_data=(test_data, test_labels),
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in history.history.keys():
                history.history[key].extend(history_finetune.history[key])
        
        # Save training history
        self.history['corrosion_detector'] = history.history
        
        # Save final model
        model.save(os.path.join(self.models_dir, f"{model_name}_final.h5"))
        
        # Evaluate model
        self.evaluate_corrosion_detector(model, test_data, test_labels)
        
        return model, history
    
    def evaluate_steel_classifier(self, model, X_test, y_test):
        """
        Evaluate steel classification model.
        """
        if not HAS_TENSORFLOW:
            return
            
        print("\n" + "="*50)
        print("STEEL CLASSIFIER EVALUATION")
        print("="*50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=['Non-Steel', 'Steel'])
        print("\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Steel', 'Steel'],
                   yticklabels=['Non-Steel', 'Steel'])
        plt.title('Steel Classifier Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'steel_classifier_confusion_matrix.png'))
        plt.show()
    
    def evaluate_corrosion_detector(self, model, test_data, test_labels):
        """
        Evaluate corrosion detection model.
        """
        if not HAS_TENSORFLOW:
            return
            
        print("\n" + "="*50)
        print("CORROSION DETECTOR EVALUATION")
        print("="*50)
        
        # Make predictions
        predictions = model.predict(test_data)
        
        if isinstance(predictions, list):
            y_pred_class, y_pred_bbox = predictions
        else:
            y_pred_class = predictions
            y_pred_bbox = None
        
        # Classification evaluation
        y_pred_classes = np.argmax(y_pred_class, axis=1)
        y_true_classes = np.argmax(test_labels['classification'], axis=1)
        
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=['No Corrosion', 'Corrosion'])
        print("\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Corrosion', 'Corrosion'],
                   yticklabels=['No Corrosion', 'Corrosion'])
        plt.title('Corrosion Detector Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'corrosion_detector_confusion_matrix.png'))
        plt.show()
        
        # Bounding box evaluation (if available)
        if y_pred_bbox is not None:
            bbox_mae = np.mean(np.abs(test_labels['regression'] - y_pred_bbox))
            print(f"\nBounding Box MAE: {bbox_mae:.4f}")
    
    def plot_training_history(self):
        """
        Plot training history for all trained models.
        """
        if not self.history:
            print("No training history available.")
            return
        
        for model_name, history in self.history.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{model_name.replace("_", " ").title()} Training History')
            
            # Plot training & validation loss
            if 'loss' in history:
                axes[0, 0].plot(history['loss'], label='Training Loss')
                axes[0, 0].plot(history['val_loss'], label='Validation Loss')
                axes[0, 0].set_title('Model Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
            
            # Plot accuracy (if available)
            if 'accuracy' in history:
                axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
                axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
                axes[0, 1].set_title('Model Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
            
            # Plot classification accuracy (for multi-output models)
            if 'classification_accuracy' in history:
                axes[1, 0].plot(history['classification_accuracy'], label='Training Accuracy')
                axes[1, 0].plot(history['val_classification_accuracy'], label='Validation Accuracy')
                axes[1, 0].set_title('Classification Accuracy')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].legend()
            
            # Plot regression MAE (for multi-output models)
            if 'regression_mae' in history:
                axes[1, 1].plot(history['regression_mae'], label='Training MAE')
                axes[1, 1].plot(history['val_regression_mae'], label='Validation MAE')
                axes[1, 1].set_title('Regression MAE')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('MAE')
                axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.models_dir, f'{model_name}_training_history.png'))
            plt.show()
    
    def save_training_config(self, config):
        """
        Save training configuration to JSON file.
        """
        config['timestamp'] = datetime.now().isoformat()
        config_path = os.path.join(self.models_dir, 'training_config.json')
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Training configuration saved to {config_path}")


def train_corrosion_model(data_dir="corrosion detect", models_dir="models", 
                         train_steel_classifier=True, train_corrosion_detector=True):
    """
    Main training function that trains both models.
    """
    if not HAS_TENSORFLOW:
        print("TensorFlow is required for training. Please install it using:")
        print("pip install tensorflow")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(data_dir, models_dir)
    
    # Training configuration
    config = {
        'data_dir': data_dir,
        'models_dir': models_dir,
        'train_steel_classifier': train_steel_classifier,
        'train_corrosion_detector': train_corrosion_detector
    }
    
    trained_models = {}
    
    # Train steel classifier
    if train_steel_classifier:
        print("\n" + "="*60)
        print("TRAINING STEEL CLASSIFIER")
        print("="*60)
        
        steel_model, steel_history = trainer.train_steel_classifier(
            epochs=30, batch_size=32, create_synthetic=True
        )
        trained_models['steel_classifier'] = steel_model
        config['steel_classifier'] = {
            'epochs': 30,
            'batch_size': 32,
            'architecture': 'Custom CNN'
        }
    
    # Train corrosion detector
    if train_corrosion_detector:
        print("\n" + "="*60)
        print("TRAINING CORROSION DETECTOR")
        print("="*60)
        
        corrosion_model, corrosion_history = trainer.train_corrosion_detector(
            epochs=50, batch_size=16, use_transfer_learning=True
        )
        trained_models['corrosion_detector'] = corrosion_model
        config['corrosion_detector'] = {
            'epochs': 50,
            'batch_size': 16,
            'architecture': 'Transfer Learning (EfficientNetB0)'
        }
    
    # Plot training histories
    trainer.plot_training_history()
    
    # Save configuration
    trainer.save_training_config(config)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Models saved in: {models_dir}")
    print("You can now use the predict.py script to make predictions.")
    
    return trained_models


if __name__ == "__main__":
    # Check if data directory exists
    data_dir = "corrosion detect"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        print("Please ensure your dataset is in the correct location.")
    else:
        # Start training
        models = train_corrosion_model(
            data_dir=data_dir,
            models_dir="models",
            train_steel_classifier=True,
            train_corrosion_detector=True
        )
