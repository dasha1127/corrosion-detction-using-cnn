"""
Utility functions for corrosion detection project.
Contains helper functions for data manipulation, evaluation metrics, and visualization.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        float: IoU score
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        intersection = 0
    else:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    # Calculate IoU
    if union == 0:
        return 0
    
    return intersection / union


def yolo_to_xyxy(yolo_bbox, img_width, img_height):
    """
    Convert YOLO format bbox to x1, y1, x2, y2 format.
    
    Args:
        yolo_bbox: [x_center, y_center, width, height] (normalized)
        img_width, img_height: Image dimensions
    
    Returns:
        list: [x1, y1, x2, y2] in pixel coordinates
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert to pixel coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Calculate corner coordinates
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    
    return [x1, y1, x2, y2]


def xyxy_to_yolo(xyxy_bbox, img_width, img_height):
    """
    Convert x1, y1, x2, y2 format bbox to YOLO format.
    
    Args:
        xyxy_bbox: [x1, y1, x2, y2] in pixel coordinates
        img_width, img_height: Image dimensions
    
    Returns:
        list: [x_center, y_center, width, height] (normalized)
    """
    x1, y1, x2, y2 = xyxy_bbox
    
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]


def calculate_map(true_boxes, pred_boxes, pred_scores, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        true_boxes: List of ground truth boxes for each image
        pred_boxes: List of predicted boxes for each image
        pred_scores: List of prediction scores for each image
        iou_threshold: IoU threshold for positive detection
        score_threshold: Score threshold for considering a detection
    
    Returns:
        float: mAP score
    """
    all_precisions = []
    all_recalls = []
    
    for i in range(len(true_boxes)):
        gt_boxes = true_boxes[i]
        pred_boxes_img = pred_boxes[i]
        pred_scores_img = pred_scores[i]
        
        # Filter predictions by score threshold
        valid_preds = pred_scores_img >= score_threshold
        pred_boxes_filtered = [pred_boxes_img[j] for j in range(len(pred_boxes_img)) if valid_preds[j]]
        pred_scores_filtered = pred_scores_img[valid_preds]
        
        # Sort by confidence
        sorted_indices = np.argsort(pred_scores_filtered)[::-1]
        pred_boxes_sorted = [pred_boxes_filtered[j] for j in sorted_indices]
        
        # Calculate precision and recall
        tp = 0
        fp = 0
        matched_gt = set()
        
        for pred_box in pred_boxes_sorted:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
    
    return np.mean(all_precisions), np.mean(all_recalls)


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot comparison of different models' metrics.
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        save_path: Path to save the plot
    """
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Extract metric values
    metric_values = {metric: [] for metric in metrics}
    
    for model in models:
        for metric in metrics:
            if metric in metrics_dict[model]:
                metric_values[metric].append(metrics_dict[model][metric])
            else:
                metric_values[metric].append(0)
    
    # Create subplot for each metric
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        axes[i].bar(models, metric_values[metric])
        axes[i].set_title(f'{metric.capitalize()}')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, v in enumerate(metric_values[metric]):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    
    plt.show()


def create_dataset_report(images_dir, labels_dir, output_file="dataset_report.json"):
    """
    Create a comprehensive report about the dataset.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        output_file: Output file for the report
    
    Returns:
        dict: Dataset statistics
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "images_directory": images_dir,
        "labels_directory": labels_dir
    }
    
    # Count images
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    report["total_images"] = len(image_files)
    
    # Analyze labels
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    report["total_label_files"] = len(label_files)
    
    # Count images with and without annotations
    images_with_labels = 0
    images_without_labels = 0
    total_annotations = 0
    bbox_areas = []
    
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                num_annotations = len([line for line in lines if line.strip()])
                
                if num_annotations > 0:
                    images_with_labels += 1
                    total_annotations += num_annotations
                    
                    # Calculate bounding box areas
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) == 5:
                                width = float(parts[3])
                                height = float(parts[4])
                                area = width * height
                                bbox_areas.append(area)
                else:
                    images_without_labels += 1
        else:
            images_without_labels += 1
    
    report["images_with_corrosion"] = images_with_labels
    report["images_without_corrosion"] = images_without_labels
    report["total_annotations"] = total_annotations
    report["avg_annotations_per_image"] = total_annotations / len(image_files) if len(image_files) > 0 else 0
    
    if bbox_areas:
        report["bbox_statistics"] = {
            "mean_area": float(np.mean(bbox_areas)),
            "std_area": float(np.std(bbox_areas)),
            "min_area": float(np.min(bbox_areas)),
            "max_area": float(np.max(bbox_areas)),
            "median_area": float(np.median(bbox_areas))
        }
    
    # Image format analysis
    format_counts = {}
    for img_file in image_files:
        ext = os.path.splitext(img_file)[1].lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1
    
    report["image_formats"] = format_counts
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Dataset report saved to {output_file}")
    
    # Print summary
    print("\n" + "="*40)
    print("DATASET REPORT SUMMARY")
    print("="*40)
    print(f"Total Images: {report['total_images']}")
    print(f"Images with Corrosion: {report['images_with_corrosion']}")
    print(f"Images without Corrosion: {report['images_without_corrosion']}")
    print(f"Total Annotations: {report['total_annotations']}")
    print(f"Average Annotations per Image: {report['avg_annotations_per_image']:.2f}")
    
    if 'bbox_statistics' in report:
        print(f"Average Bounding Box Area: {report['bbox_statistics']['mean_area']:.4f}")
    
    return report


def augment_single_image(image, bbox=None, augmentation_type='random'):
    """
    Apply augmentation to a single image and adjust bounding box accordingly.
    
    Args:
        image: Input image (numpy array)
        bbox: Bounding box in YOLO format [x_center, y_center, width, height]
        augmentation_type: Type of augmentation to apply
    
    Returns:
        tuple: (augmented_image, adjusted_bbox)
    """
    h, w = image.shape[:2]
    
    if augmentation_type == 'horizontal_flip':
        # Flip image horizontally
        augmented_image = cv2.flip(image, 1)
        
        if bbox is not None:
            x_center, y_center, width, height = bbox
            # Flip x coordinate
            x_center = 1.0 - x_center
            adjusted_bbox = [x_center, y_center, width, height]
        else:
            adjusted_bbox = None
            
    elif augmentation_type == 'brightness':
        # Adjust brightness
        factor = np.random.uniform(0.7, 1.3)
        augmented_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        adjusted_bbox = bbox
        
    elif augmentation_type == 'rotation':
        # Rotate image (small angle)
        angle = np.random.uniform(-15, 15)
        center = (w//2, h//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # For simplicity, keep bbox unchanged (in practice, should transform bbox)
        adjusted_bbox = bbox
        
    else:  # random
        # Randomly choose an augmentation
        aug_types = ['horizontal_flip', 'brightness', 'rotation']
        chosen_type = np.random.choice(aug_types)
        return augment_single_image(image, bbox, chosen_type)
    
    return augmented_image, adjusted_bbox


def visualize_dataset_distribution(images_dir, labels_dir, save_path=None):
    """
    Visualize the distribution of the dataset.
    """
    # Get dataset statistics
    report = create_dataset_report(images_dir, labels_dir)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Corrosion vs No Corrosion
    labels = ['With Corrosion', 'Without Corrosion']
    sizes = [report['images_with_corrosion'], report['images_without_corrosion']]
    colors = ['#ff6b6b', '#4ecdc4']
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Corrosion Distribution')
    
    # 2. Image Formats
    formats = list(report['image_formats'].keys())
    counts = list(report['image_formats'].values())
    
    axes[0, 1].bar(formats, counts, color='skyblue')
    axes[0, 1].set_title('Image Format Distribution')
    axes[0, 1].set_ylabel('Count')
    
    # 3. Bounding Box Area Distribution
    if 'bbox_statistics' in report:
        # This would need actual bbox data to create histogram
        axes[1, 0].text(0.5, 0.5, 'Bounding Box Area\nDistribution\n(Need actual data)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Bbox Area Distribution')
    
    # 4. Annotations per Image
    axes[1, 1].bar(['Avg Annotations'], [report['avg_annotations_per_image']], color='lightcoral')
    axes[1, 1].set_title('Average Annotations per Image')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset distribution visualization saved to {save_path}")
    
    plt.show()


def create_model_comparison_report(models_results, output_file="model_comparison.json"):
    """
    Create a comparison report for different models.
    
    Args:
        models_results: Dictionary with model names and their evaluation results
        output_file: Output file for the report
    
    Returns:
        dict: Comparison report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "models_compared": list(models_results.keys()),
        "comparison": {}
    }
    
    for model_name, results in models_results.items():
        report["comparison"][model_name] = {
            "accuracy": results.get("accuracy", 0),
            "precision": results.get("precision", 0),
            "recall": results.get("recall", 0),
            "f1_score": results.get("f1_score", 0),
            "training_time": results.get("training_time", "N/A"),
            "model_size": results.get("model_size", "N/A"),
            "inference_time": results.get("inference_time", "N/A")
        }
    
    # Find best model for each metric
    best_models = {}
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        best_score = -1
        best_model = ""
        for model_name, metrics in report["comparison"].items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name
        best_models[metric] = {"model": best_model, "score": best_score}
    
    report["best_models"] = best_models
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Model comparison report saved to {output_file}")
    
    return report


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test IoU calculation
    box1 = [10, 10, 50, 50]
    box2 = [30, 30, 70, 70]
    iou = calculate_iou(box1, box2)
    print(f"IoU between {box1} and {box2}: {iou:.3f}")
    
    # Test coordinate conversion
    yolo_bbox = [0.5, 0.5, 0.4, 0.3]
    img_w, img_h = 224, 224
    xyxy_bbox = yolo_to_xyxy(yolo_bbox, img_w, img_h)
    print(f"YOLO {yolo_bbox} -> XYXY {xyxy_bbox}")
    
    converted_back = xyxy_to_yolo(xyxy_bbox, img_w, img_h)
    print(f"XYXY {xyxy_bbox} -> YOLO {converted_back}")
    
    print("Utility functions test completed successfully!")
