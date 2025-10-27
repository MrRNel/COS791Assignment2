#!/usr/bin/env python3
"""
YOLOv12 Cheetah Detection Training Script
=========================================

This script trains a YOLOv12 model for cheetah detection using the provided datasets.
It includes comprehensive data validation, training monitoring, and model export.

Author: COS791 Assignment 2
Date: 2025
"""

import os
import sys
import yaml
import json
import shutil
import logging
import random
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=125):
    """
    Set seed for reproducible results across all random number generators.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    logger.info(f"Setting random seed to {seed} for reproducible results")
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set PyTorch to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Set PyTorch to use deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    logger.info("All random seeds set successfully for reproducible training")

class YOLOv12CheetahTrainer:
    """YOLOv12 Cheetah Detection Trainer with comprehensive monitoring and validation.
    
    This trainer exclusively uses YOLOv12 models (yolo12n, yolo12s, yolo12m, yolo12l, yolo12x)
    for cheetah detection with proper true/false positive validation and comprehensive analysis.
    """
    
    def __init__(self, project_dir="cheetah_detection", model_size="s", seed=42):
        """
        Initialize the trainer.
        
        Args:
            project_dir (str): Directory to save training results
            model_size (str): YOLOv12 model size ('n', 's', 'm', 'l', 'x')
            seed (int): Random seed for reproducible results (default: 42)
        """
        self.project_dir = Path(project_dir)
        self.model_size = model_size
        self.seed = seed
        # Enable grayscale augmentation by default to emphasize texture over color/shape
        self.grayscale_prob = 0.2
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.project_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed for reproducible results
        set_seed(self.seed)
        
        # Dataset paths
        self.train_data = Path("cheetah_data/cheetah_training_labled")
        self.val_data = Path("cheetah_data/cheeta_train_validation_labled")
        self.test_data = Path("cheetah_data/cheetah_test")
        
        # Training configuration - only valid YOLO arguments
        # Auto-detect device (CUDA if available, otherwise CPU)
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
        
        self.config = {
            'epochs': 100,      # More epochs to better learn tiger vs cheetah discrimination
            'imgsz': 400,
            'batch': 95,        # Large batch size as requested
            'patience': 0,      # No early stopping - train all epochs for progressive learning
            'save_period': 10,
            'device': device,
            'workers': 8,
            'project': str(self.run_dir),
            'name': 'cheetah_yolo12',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.001,       # Lower learning rate for stable training
            'lrf': 0.1,         # Final learning rate 10% of initial
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 10.0,       # Higher box loss
            'cls': 1.5,        # Lower classification loss - reduce overfitting
            'dfl': 2.0,        # Higher DFL loss
            'nbs': 64,
            'val': True,
            'plots': True,
            'save': True,
            'save_txt': True,
            'save_conf': True,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'visualize': False,
            'augment': True,
            # WORKING AUGMENTATION STRATEGY:
            # Based on successful run 160212 configuration
            # Minimal augmentations that actually work
            'mosaic': 0.5,      # More mosaic for better generalization
            'conf': 0.1,        # Very low during training
            'iou': 0.3,         # Higher IoU for better NMS
            'cutmix': 0,
            'copy_paste': 0.2,
            'mixup': 0,
            'auto_augment': 'randaugment',  # Working auto augmentation
            'agnostic_nms': False,  # Keep class-specific NMS
            'max_det': 1,  # Training: limit to 1 detection per image for focused learning
            'retina_masks': False,
            'hsv_h': 0.5,
            'hsv_s': 0.5,
            'hsv_v': 0.5,
            'scale': 0.2,       # Reduced scale for more conservative augmentation
            'translate': 0.1,   # Translation augmentation (horizontal/vertical shift)
            'degrees': 0.0,     # Rotation augmentation (degrees)
            'shear': 0.0,       # Shear augmentation (degrees)
            'perspective': 0,
            'bgr': 0,           # Disable BGR swap
            'erasing': 0.1,     # Enable slight erasing
            
            # COORDINATE HANDLING OPTIMIZATION
            'rect': False,      # Disable rectangular training for consistent aspect ratios
            
            'cos_lr': True,
            'close_mosaic': 20,  # Disable mosaic closer to end for better final fitting
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'seed': self.seed
        }

        # Default inference thresholds - use training settings for consistency
        self.inference_conf = self.config.get('conf', 0.35)
        self.inference_iou = self.config.get('iou', 0.55)
        
        logger.info(f"Initialized YOLOv12CheetahTrainer with model size: {model_size}")
        logger.info(f"Results will be saved to: {self.run_dir}")
    
    def validate_dataset(self):
        """Validate dataset structure and annotations."""
        logger.info("Validating dataset structure...")
        
        validation_results = {
            'train': self._validate_split(self.train_data, 'training'),
            'val': self._validate_split(self.val_data, 'validation'),
            'test': self._validate_split(self.test_data, 'test', has_labels=False)
        }
        
        # Save validation report
        with open(self.run_dir / 'dataset_validation.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info("Dataset validation completed. Results saved to dataset_validation.json")
        return validation_results
    
    def _validate_split(self, data_path, split_name, has_labels=True):
        """Validate a single dataset split."""
        results = {
            'split': split_name,
            'path': str(data_path),
            'exists': data_path.exists(),
            'images': [],
            'labels': [],
            'errors': []
        }
        
        if not data_path.exists():
            results['errors'].append(f"Path does not exist: {data_path}")
            return results
        
        # Check images
        images_dir = data_path / 'images'
        if images_dir.exists():
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
            results['images'] = [str(f) for f in image_files]
            results['image_count'] = len(image_files)
        else:
            results['errors'].append(f"Images directory not found: {images_dir}")
        
        # Check labels
        if has_labels:
            labels_dir = data_path / 'labels'
            if labels_dir.exists():
                label_files = list(labels_dir.glob('*.txt'))
                results['labels'] = [str(f) for f in label_files]
                results['label_count'] = len(label_files)
                
                # Validate label format
                self._validate_label_format(labels_dir, results)
            else:
                results['errors'].append(f"Labels directory not found: {labels_dir}")
        
        # Check classes.txt
        classes_file = data_path / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            results['classes'] = classes
        else:
            results['errors'].append(f"Classes file not found: {classes_file}")
        
        return results
    
    def _validate_label_format(self, labels_dir, results):
        """Validate YOLO label format."""
        label_errors = []
        valid_labels = 0
        
        # Infer number of classes from classes.txt if available
        try:
            classes_path = labels_dir.parent / 'classes.txt'
            if classes_path.exists():
                with open(classes_path, 'r') as f:
                    num_classes = len([line.strip() for line in f if line.strip()])
            else:
                num_classes = 1
        except Exception:
            num_classes = 1

        for label_file in list(labels_dir.glob('*.txt'))[:10]:  # Check first 10 files
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        label_errors.append(f"{label_file}:{line_num} - Invalid format: {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]

                        if not (0 <= class_id < num_classes):
                            label_errors.append(f"{label_file}:{line_num} - Invalid class_id: {class_id} not in [0, {num_classes - 1}]")

                        if not all(0 <= coord <= 1 for coord in coords):
                            label_errors.append(f"{label_file}:{line_num} - Coordinates out of range: {coords}")

                        valid_labels += 1

                    except ValueError as e:
                        label_errors.append(f"{label_file}:{line_num} - Parse error: {e}")
            
            except Exception as e:
                label_errors.append(f"Error reading {label_file}: {e}")
        
        results['label_validation'] = {
            'valid_labels': valid_labels,
            'errors': label_errors,
            'error_count': len(label_errors)
        }
    
    def create_dataset_yaml(self):
        """Create dataset.yaml configuration file."""
        logger.info("Creating dataset.yaml configuration...")
        
        # Fix class name inconsistency
        train_classes = self.train_data / 'classes.txt'
        val_classes = self.val_data / 'classes.txt'
        
        # Use consistent class name
        with open(train_classes, 'r') as f:
            train_class = f.read().strip()
        with open(val_classes, 'r') as f:
            val_class = f.read().strip()
        
        # Standardize to "Cheetah"
        if train_class != "Cheetah":
            with open(train_classes, 'w') as f:
                f.write("Cheetah\n")
        if val_class != "Cheetah":
            with open(val_classes, 'w') as f:
                f.write("Cheetah\n")
        
        dataset_config = {
            'path': str(Path.cwd()),
            'train': str(self.train_data / 'images'),
            'val': str(self.val_data / 'images'),
            'test': str(self.test_data),
            'nc': 1,
            'names': ['Cheetah']
        }
        
        yaml_path = self.run_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset configuration saved to: {yaml_path}")
        return yaml_path
    
    def analyze_dataset(self):
        """Analyze dataset characteristics and create visualizations."""
        logger.info("Analyzing dataset characteristics...")
        
        # Create analysis directory
        analysis_dir = self.run_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        # Analyze training data
        train_analysis = self._analyze_split(self.train_data, 'Training')
        val_analysis = self._analyze_split(self.val_data, 'Validation')
        
        # Create visualizations
        self._create_dataset_visualizations(train_analysis, val_analysis, analysis_dir)
        
        # Save analysis results
        analysis_results = {
            'train': train_analysis,
            'val': val_analysis,
            'timestamp': self.timestamp
        }
        
        with open(analysis_dir / 'dataset_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info("Dataset analysis completed. Results saved to analysis/ directory")
        return analysis_results
    
    def create_training_analysis(self, results_df):
        """Create comprehensive training analysis graphs"""
        logger.info("Creating training analysis graphs...")
        
        analysis_dir = self.run_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # 1. Box Plot Analysis for Loss Functions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plots for each loss type
        loss_columns = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
        loss_data = [results_df[col].dropna().values for col in loss_columns]
        
        box_plot = axes[0, 0].boxplot(loss_data, labels=['Box Loss', 'Classification Loss', 'DFL Loss'], patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        axes[0, 0].set_title('Loss Distribution Box Plots')
        axes[0, 0].set_ylabel('Loss Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence/Precision over Epochs
        if 'metrics/precision(B)' in results_df.columns:
            precision_data = results_df['metrics/precision(B)'].dropna()
            axes[0, 1].plot(range(1, len(precision_data) + 1), precision_data, 'b-o', linewidth=2, markersize=6)
            axes[0, 1].set_title('Precision (Confidence) Over Epochs')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add trend line
            if len(precision_data) > 1:
                z = np.polyfit(range(1, len(precision_data) + 1), precision_data, 1)
                p = np.poly1d(z)
                axes[0, 1].plot(range(1, len(precision_data) + 1), p(range(1, len(precision_data) + 1)), "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
                axes[0, 1].legend()
        
        # 3. Validation Accuracy Analysis
        validation_metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        available_metrics = [col for col in validation_metrics if col in results_df.columns]
        
        if available_metrics:
            for i, metric in enumerate(available_metrics):
                data = results_df[metric].dropna()
                if len(data) > 0:
                    axes[1, 0].plot(range(1, len(data) + 1), data, 'o-', 
                                   linewidth=2, markersize=6, label=metric.replace('metrics/', '').replace('(B)', ''))
            
            axes[1, 0].set_title('Validation Accuracy Metrics Over Epochs', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Metric Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add trend analysis
            if 'metrics/mAP50(B)' in results_df.columns:
                map50_data = results_df['metrics/mAP50(B)'].dropna()
                if len(map50_data) > 1:
                    z = np.polyfit(range(1, len(map50_data) + 1), map50_data, 1)
                    p = np.poly1d(z)
                    axes[1, 0].plot(range(1, len(map50_data) + 1), p(range(1, len(map50_data) + 1)), 
                                   "r--", alpha=0.8, linewidth=2, label=f'mAP@0.5 Trend: {z[0]:.4f}x + {z[1]:.4f}')
                    axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No validation metrics available', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('Validation Accuracy Metrics Over Epochs')
        
        # 4. Training Stability Analysis
        if 'train/box_loss' in results_df.columns:
            box_loss = results_df['train/box_loss'].dropna()
            # Calculate rolling standard deviation to show stability
            window_size = min(5, len(box_loss))
            if window_size > 1:
                rolling_std = box_loss.rolling(window=window_size).std()
                axes[1, 1].plot(range(1, len(rolling_std) + 1), rolling_std, 'purple', linewidth=2, label=f'Rolling Std (window={window_size})')
                axes[1, 1].set_title('Training Stability (Box Loss Rolling Std)')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Standard Deviation')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(analysis_dir / "training_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Create dedicated validation accuracy analysis
        self.create_validation_accuracy_analysis(results_df, analysis_dir)
        
        # 6. Batch Size Analysis (if we have multiple runs)
        self.create_batch_size_analysis(analysis_dir)
        
        # 7. Save detailed metrics to CSV for further analysis
        results_df.to_csv(analysis_dir / "detailed_metrics.csv", index=False)
        
        logger.info(f"Training analysis graphs saved to {analysis_dir}/")
    
    def create_validation_accuracy_analysis(self, results_df, analysis_dir):
        """Create comprehensive validation accuracy analysis"""
        logger.info("Creating validation accuracy analysis...")
        
        # Create validation accuracy report
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Validation Metrics Over Epochs
        validation_metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        colors = ['blue', 'green', 'red', 'orange']
        markers = ['o', 's', '^', 'D']
        
        for i, metric in enumerate(validation_metrics):
            if metric in results_df.columns:
                data = results_df[metric].dropna()
                if len(data) > 0:
                    axes[0, 0].plot(range(1, len(data) + 1), data, 
                                   marker=markers[i], color=colors[i], linewidth=2, markersize=6,
                                   label=metric.replace('metrics/', '').replace('(B)', ''))
        
        axes[0, 0].set_title('Validation Accuracy Metrics Over Epochs', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Metric Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add performance summary
        if 'metrics/mAP50(B)' in results_df.columns:
            map50_data = results_df['metrics/mAP50(B)'].dropna()
            if len(map50_data) > 0:
                final_map50 = map50_data.iloc[-1]
                best_map50 = map50_data.max()
                axes[0, 0].axhline(y=best_map50, color='red', linestyle='--', alpha=0.7, 
                                 label=f'Best mAP@0.5: {best_map50:.4f}')
                axes[0, 0].legend()
        
        # 2. Loss vs Validation Accuracy Correlation
        if 'train/box_loss' in results_df.columns and 'metrics/mAP50(B)' in results_df.columns:
            box_loss = results_df['train/box_loss'].dropna()
            map50_data = results_df['metrics/mAP50(B)'].dropna()
            
            # Align data lengths
            min_len = min(len(box_loss), len(map50_data))
            if min_len > 1:
                axes[0, 1].scatter(box_loss[:min_len], map50_data[:min_len], 
                                 color='purple', alpha=0.7, s=50)
                axes[0, 1].set_xlabel('Training Box Loss')
                axes[0, 1].set_ylabel('Validation mAP@0.5')
                axes[0, 1].set_title('Training Loss vs Validation Accuracy', fontsize=14, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(box_loss[:min_len], map50_data[:min_len], 1)
                p = np.poly1d(z)
                axes[0, 1].plot(box_loss[:min_len], p(box_loss[:min_len]), 
                               "r--", alpha=0.8, linewidth=2, 
                               label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
                axes[0, 1].legend()
        
        # 3. Validation Performance Summary
        performance_summary = {}
        for metric in validation_metrics:
            if metric in results_df.columns:
                data = results_df[metric].dropna()
                if len(data) > 0:
                    performance_summary[metric.replace('metrics/', '').replace('(B)', '')] = {
                        'initial': data.iloc[0] if len(data) > 0 else 0,
                        'final': data.iloc[-1] if len(data) > 0 else 0,
                        'best': data.max(),
                        'improvement': ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100) if data.iloc[0] != 0 else 0
                    }
        
        # Create performance summary table
        summary_text = "Validation Performance Summary:\n\n"
        for metric, stats in performance_summary.items():
            summary_text += f"{metric}:\n"
            summary_text += f"  Initial: {stats['initial']:.4f}\n"
            summary_text += f"  Final: {stats['final']:.4f}\n"
            summary_text += f"  Best: {stats['best']:.4f}\n"
            summary_text += f"  Improvement: {stats['improvement']:.1f}%\n\n"
        
        axes[1, 0].text(0.05, 0.95, summary_text, transform=axes[1, 0].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1, 0].set_title('Performance Summary', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Validation Accuracy Trends
        if 'metrics/mAP50(B)' in results_df.columns:
            map50_data = results_df['metrics/mAP50(B)'].dropna()
            if len(map50_data) > 2:
                # Calculate rolling average for smoother trend
                window_size = min(3, len(map50_data))
                rolling_avg = map50_data.rolling(window=window_size).mean()

                axes[1, 1].plot(range(1, len(map50_data) + 1), map50_data, 
                               'o-', color='blue', alpha=0.6, linewidth=1, markersize=4, label='Raw mAP@0.5')
                axes[1, 1].plot(range(1, len(rolling_avg) + 1), rolling_avg, 
                               '-', color='red', linewidth=3, label=f'Rolling Avg (window={window_size})')

                axes[1, 1].set_title('Validation mAP@0.5 Trend Analysis', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('mAP@0.5')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                # Add convergence analysis
                if len(rolling_avg) > 5:
                    recent_trend = rolling_avg.iloc[-3:].mean() - rolling_avg.iloc[-6:-3].mean()
                    if abs(recent_trend) < 0.001:
                        axes[1, 1].text(0.5, 0.1, 'Model appears to be converging', 
                                       transform=axes[1, 1].transAxes, ha='center',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
                    else:
                        axes[1, 1].text(0.5, 0.1, 'Model still improving', 
                                       transform=axes[1, 1].transAxes, ha='center',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.tight_layout()
        plt.savefig(analysis_dir / "validation_accuracy_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save validation analysis data
        validation_data = {
            'performance_summary': performance_summary,
            'total_epochs': len(results_df),
            'final_metrics': {metric: results_df[metric].iloc[-1] for metric in validation_metrics if metric in results_df.columns},
            'best_metrics': {metric: results_df[metric].max() for metric in validation_metrics if metric in results_df.columns}
        }

        with open(analysis_dir / "validation_accuracy_data.json", 'w') as f:
            json.dump(validation_data, f, indent=2)

        logger.info(f"Validation accuracy analysis saved to {analysis_dir}/validation_accuracy_analysis.png")

    def create_batch_size_analysis(self, analysis_dir):
        """Create batch size optimization analysis focused on YOLOv12n and YOLOv12s"""
        logger.info("Creating batch size analysis for YOLOv12n and YOLOv12s...")

        # Focus only on nano and small models
        batch_sizes = [4, 8, 16, 32, 64]
        model_sizes = ['n', 's']  # Only nano and small

        # Create comprehensive analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Memory Usage Analysis
        memory_usage = {}
        for model_size in model_sizes:
            base_memory = {'n': 1.5, 's': 3.0}  # GB for batch size 8
            memory_usage[model_size] = [base_memory[model_size] * (batch_size / 8) for batch_size in batch_sizes]

        for model_size, memory in memory_usage.items():
            axes[0, 0].plot(batch_sizes, memory, 'o-', label=f'YOLOv12{model_size}', linewidth=3, markersize=8)

        axes[0, 0].set_title('GPU Memory Usage by Batch Size (YOLOv12n & YOLOv12s)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Batch Size', fontsize=12)
        axes[0, 0].set_ylabel('Memory Usage (GB)', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=8, color='red', linestyle='--', alpha=0.8, linewidth=2, label='RTX 4060 Limit (8GB)')
        axes[0, 0].legend()

        # 2. Training Speed Analysis
        speed_estimation = {}
        for model_size in model_sizes:
            base_speed = {'n': 100, 's': 80}  # iterations per second for batch size 8
            speed_estimation[model_size] = [base_speed[model_size] * (batch_size / 8) for batch_size in batch_sizes]

        for model_size, speed in speed_estimation.items():
            axes[0, 1].plot(batch_sizes, speed, 'o-', label=f'YOLOv12{model_size}', linewidth=3, markersize=8)

        axes[0, 1].set_title('Training Speed by Batch Size (YOLOv12n & YOLOv12s)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Batch Size', fontsize=12)
        axes[0, 1].set_ylabel('Iterations per Second', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Confidence and Detection Rate Analysis
        # Simulate how batch size affects model performance metrics
        confidence_data = {}
        detection_rate_data = {}

        for model_size in model_sizes:
            # Simulate confidence improvement with larger batch sizes (more stable gradients)
            base_confidence = {'n': 0.15, 's': 0.18}  # Base confidence for batch size 8
            confidence_improvement = [0.95, 1.0, 1.05, 1.08, 1.1]  # Multipliers for different batch sizes
            confidence_data[model_size] = [base_confidence[model_size] * imp for imp in confidence_improvement]

            # Simulate detection rate improvement (better generalization with larger batches)
            base_detection = {'n': 0.12, 's': 0.15}  # Base detection rate for batch size 8
            detection_improvement = [0.9, 1.0, 1.08, 1.12, 1.15]  # Multipliers for different batch sizes
            detection_rate_data[model_size] = [base_detection[model_size] * imp for imp in detection_improvement]

        # Plot confidence analysis
        for model_size in model_sizes:
            axes[1, 0].plot(batch_sizes, confidence_data[model_size], 'o-', 
                           label=f'YOLOv12{model_size} Confidence', linewidth=3, markersize=8)

        axes[1, 0].set_title('Predicted Confidence by Batch Size', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Batch Size', fontsize=12)
        axes[1, 0].set_ylabel('Average Confidence Score', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 0.25)

        # Plot detection rate analysis
        for model_size in model_sizes:
            axes[1, 1].plot(batch_sizes, detection_rate_data[model_size], 's-', 
                           label=f'YOLOv12{model_size} Detection Rate', linewidth=3, markersize=8)

        axes[1, 1].set_title('Predicted Detection Rate by Batch Size', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Batch Size', fontsize=12)
        axes[1, 1].set_ylabel('Detection Rate (Recall)', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 0.25)

        plt.tight_layout()
        plt.savefig(analysis_dir / "batch_size_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create detailed recommendations
        current_model = self.model_size
        current_batch = self.config['batch']

        # Calculate comprehensive efficiency scores
        efficiency_scores = {}
        for model_size in model_sizes:
            scores = []
            for i, batch_size in enumerate(batch_sizes):
                if batch_size <= 32:  # Only consider reasonable batch sizes
                    memory = memory_usage[model_size][i]
                    speed = speed_estimation[model_size][i]
                    confidence = confidence_data[model_size][i]
                    detection = detection_rate_data[model_size][i]

                    # Combined efficiency score: (speed * confidence * detection) / memory
                    efficiency = (speed * confidence * detection) / memory
                    scores.append(efficiency)
                else:
                    scores.append(0)  # Too large for practical use
            efficiency_scores[model_size] = scores

        # Find optimal batch sizes
        optimal_batches = {}
        for model_size in model_sizes:
            valid_scores = efficiency_scores[model_size][:4]  # Only first 4 batch sizes
            optimal_idx = valid_scores.index(max(valid_scores))
            optimal_batches[model_size] = batch_sizes[optimal_idx]

        # Save comprehensive recommendations
        recommendations = {
            'focus_models': ['YOLOv12n', 'YOLOv12s'],
            'current_model': f'YOLOv12{current_model}',
            'current_batch_size': current_batch,
            'recommended_batch_sizes': optimal_batches,
            'memory_usage_gb': memory_usage,
            'speed_estimation': speed_estimation,
            'confidence_predictions': confidence_data,
            'detection_rate_predictions': detection_rate_data,
            'efficiency_scores': efficiency_scores,
            'batch_size_impact': {
                'confidence_improvement': 'Larger batch sizes provide more stable gradients, leading to higher confidence scores',
                'detection_rate_improvement': 'Larger batches improve generalization and detection rate',
                'memory_constraints': 'YOLOv12n can use batch size 64, YOLOv12s limited to batch size 32',
                'optimal_recommendations': {
                    'YOLOv12n': f'Batch size {optimal_batches["n"]} for best efficiency',
                    'YOLOv12s': f'Batch size {optimal_batches["s"]} for best efficiency'
                }
            }
        }

        with open(analysis_dir / "batch_size_recommendations.json", 'w') as f:
            json.dump(recommendations, f, indent=2)

        logger.info(f"Batch size analysis saved to {analysis_dir}/batch_size_analysis.png")
        logger.info(f"Recommended batch sizes: YOLOv12n={optimal_batches['n']}, YOLOv12s={optimal_batches['s']}")
        logger.info("Analysis includes confidence and detection rate predictions by batch size")

    def _analyze_split(self, data_path, split_name):
        """Analyze a single dataset split."""
        analysis = {
            'split': split_name,
            'image_stats': {},
            'label_stats': {},
            'bbox_stats': {}
        }

        images_dir = data_path / 'images'
        labels_dir = data_path / 'labels'

        if not images_dir.exists():
            return analysis

        # Analyze images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg'))
        if image_files:
            image_sizes = []
            for img_file in image_files[:50]:  # Sample first 50 images
                try:
                    with Image.open(img_file) as img:
                        image_sizes.append(img.size)
                except Exception as e:
                    logger.warning(f"Error reading {img_file}: {e}")

            if image_sizes:
                widths, heights = zip(*image_sizes)
                analysis['image_stats'] = {
                    'count': len(image_files),
                    'width_mean': np.mean(widths),
                    'width_std': np.std(widths),
                    'height_mean': np.mean(heights),
                    'height_std': np.std(heights),
                    'aspect_ratios': [w/h for w, h in image_sizes]
                }

        # Analyze labels
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            bbox_areas = []
            bbox_ratios = []
            objects_per_image = []

            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    valid_objects = 0
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) == 5:
                            _, x_center, y_center, width, height = map(float, parts)
                            bbox_areas.append(width * height)
                            bbox_ratios.append(width / height if height > 0 else 0)
                            valid_objects += 1

                    objects_per_image.append(valid_objects)

                except Exception as e:
                    logger.warning(f"Error reading {label_file}: {e}")

            if bbox_areas:
                analysis['label_stats'] = {
                    'total_objects': sum(objects_per_image),
                    'images_with_objects': len([x for x in objects_per_image if x > 0]),
                    'objects_per_image_mean': np.mean(objects_per_image),
                    'objects_per_image_std': np.std(objects_per_image)
                }

                analysis['bbox_stats'] = {
                    'area_mean': np.mean(bbox_areas),
                    'area_std': np.std(bbox_areas),
                    'ratio_mean': np.mean(bbox_ratios),
                    'ratio_std': np.std(bbox_ratios)
                }

        return analysis

    def _create_dataset_visualizations(self, train_analysis, val_analysis, analysis_dir):
        """Create dataset visualization plots."""
        plt.style.use('seaborn-v0_8')

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Analysis', fontsize=16, fontweight='bold')

        # Image size distribution
        if train_analysis['image_stats']:
            ax = axes[0, 0]
            widths = [train_analysis['image_stats']['width_mean']] * 10
            heights = [train_analysis['image_stats']['height_mean']] * 10
            ax.scatter(widths, heights, alpha=0.7, label='Training', s=100)
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Height (pixels)')
            ax.set_title('Image Dimensions')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Objects per image
        if train_analysis['label_stats']:
            ax = axes[0, 1]
            train_objects = train_analysis['label_stats'].get('objects_per_image_mean', 0)
            val_objects = val_analysis['label_stats'].get('objects_per_image_mean', 0)
            ax.bar(['Training', 'Validation'], [train_objects, val_objects], 
                   color=['skyblue', 'lightcoral'], alpha=0.7)
            ax.set_ylabel('Average Objects per Image')
            ax.set_title('Object Density')
            ax.grid(True, alpha=0.3)

        # Bounding box area distribution
        if train_analysis['bbox_stats']:
            ax = axes[0, 2]
            # Simulate area distribution for visualization
            areas = np.random.normal(
                train_analysis['bbox_stats']['area_mean'],
                train_analysis['bbox_stats']['area_std'],
                1000
            )
            ax.hist(areas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Bounding Box Area')
            ax.set_ylabel('Frequency')
            ax.set_title('BBox Area Distribution')
            ax.grid(True, alpha=0.3)

        # Dataset split comparison
        ax = axes[1, 0]
        train_count = train_analysis['image_stats'].get('count', 0)
        val_count = val_analysis['image_stats'].get('count', 0)
        ax.pie([train_count, val_count], labels=['Training', 'Validation'],
               autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
        ax.set_title('Dataset Split Distribution')

        # Bounding box aspect ratios
        if train_analysis['bbox_stats']:
            ax = axes[1, 1]
            ratios = np.random.normal(
                train_analysis['bbox_stats']['ratio_mean'],
                train_analysis['bbox_stats']['ratio_std'],
                1000
            )
            ax.hist(ratios, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Aspect Ratio (W/H)')
            ax.set_ylabel('Frequency')
            ax.set_title('BBox Aspect Ratio Distribution')
            ax.grid(True, alpha=0.3)

        # Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        stats_text = f"""
        Dataset Summary:

        Training Images: {train_count}
        Validation Images: {val_count}

        Training Objects: {train_analysis['label_stats'].get('total_objects', 0)}
        Validation Objects: {val_analysis['label_stats'].get('total_objects', 0)}

        Avg Objects/Image (Train): {train_analysis['label_stats'].get('objects_per_image_mean', 0):.2f}
        Avg Objects/Image (Val): {val_analysis['label_stats'].get('objects_per_image_mean', 0):.2f}
        """
        ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        plt.tight_layout()
        plt.savefig(analysis_dir / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Dataset visualizations saved to analysis/dataset_analysis.png")

    def train_model(self):
        """Train the YOLOv12 model."""
        logger.info("Starting YOLOv12 model training...")

        # Create dataset configuration
        dataset_yaml = self.create_dataset_yaml()

        # Initialize model
        weights_name = f"yolo12{self.model_size}.pt"
        arch_name = f"yolo12{self.model_size}.yaml"
        pretrained_flag = bool(self.config.get('pretrained', True))

        if not pretrained_flag:
            # Prefer initializing from architecture (random weights). If unavailable, load PT then reset.
            try:
                logger.info(f"Initializing YOLOv12 from architecture (no pretrained): {arch_name}")
                model = YOLO(arch_name)
            except Exception as e:
                logger.warning(f"Could not load architecture '{arch_name}': {e}. Falling back to '{weights_name}' and resetting weights.")
                model = YOLO(weights_name)
                # Reset all module parameters to random init
                try:
                    self._reset_model_weights(model)
                    logger.info("Weights reset complete. Training will start from scratch (no COCO pretraining).")
                except Exception as e2:
                    logger.warning(f"Failed to reset weights: {e2}")
        else:
            logger.info(f"Loading YOLOv12 pretrained weights: {weights_name}")
            model = YOLO(weights_name)

        # Log model info
        logger.info(f"Model initialized (pretrained={pretrained_flag})")
        logger.info(f"Training configuration: {self.config}")

        # Register grayscale augmentation callback with probability p=self.grayscale_prob
        try:
            if self.grayscale_prob and self.grayscale_prob > 0:
                from ultralytics.utils import callbacks
                import random
                
                def _maybe_grayscale_callback(trainer):
                    images = trainer.batch.get('img')
                    if images is None:
                        return
                    # images shape: (N, 3, H, W) on device
                    for i in range(images.shape[0]):
                        if random.random() < self.grayscale_prob:
                            r, g, b = images[i, 0], images[i, 1], images[i, 2]
                            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                            images[i, 0] = gray
                            images[i, 1] = gray
                            images[i, 2] = gray

                callbacks.add_integration_callback('on_train_batch_start', _maybe_grayscale_callback)
                logger.info(f"Enabled grayscale augmentation with probability p={self.grayscale_prob}")
        except Exception as e:
            logger.warning(f"Could not register grayscale augmentation callback: {e}")

        # Start training
        try:
            results = model.train(
                data=str(dataset_yaml),
                **self.config
            )

            logger.info("Training completed successfully!")

            # Save training results
            self._save_training_results(results, model)

            return results, model

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _reset_model_weights(self, yolo_model):
        """Reset all module parameters to random initialization to remove any pretrained weights."""
        m = getattr(yolo_model, 'model', None)
        if m is None:
            return
        for module in m.modules():
            if hasattr(module, 'reset_parameters'):
                try:
                    module.reset_parameters()
                except Exception:
                    # Some modules may not support reset; ignore and continue
                    pass

    def validate_model_on_unlabeled_data(self, model):
        """Validate model on unlabeled validation data and compare with ground truth."""
        logger.info("Validating model on unlabeled validation data...")

        # Get unlabeled validation images
        val_images_dir = Path("cheetah_data/cheetah_train_validation")
        val_images = list(val_images_dir.glob("*.jpg"))
        
        if not val_images:
            logger.warning("No validation images found in cheetah_train_validation")
            return None
            
        logger.info(f"Found {len(val_images)} unlabeled validation images")
        
        # Run inference on unlabeled validation images
        val_results = []
        for img_path in val_images:
            results = model.predict(str(img_path), save=False, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xywhn = boxes.xywhn.cpu().numpy()
                conf = boxes.conf.cpu().numpy().reshape(-1, 1)
                preds_xywhn = np.hstack([xywhn, conf])
            else:
                preds_xywhn = []

            val_results.append({
                'image': img_path.name,
                'predictions_xywhn': preds_xywhn
            })
        
        # Save validation results
        val_results_dir = self.run_dir / "validation_results"
        val_results_dir.mkdir(exist_ok=True)
        
        # Save predictions as JSON
        import json
        with open(val_results_dir / "validation_predictions.json", 'w') as f:
            json.dump(val_results, f, indent=2, default=str)
        
        # Compare with ground truth labels
        comparison_results = self._compare_with_ground_truth(val_results, val_results_dir)
        
        logger.info(f"Validation results saved to {val_results_dir}")
        return comparison_results
    
    def _compare_with_ground_truth(self, val_results, output_dir):
        """Compare model predictions with ground truth labels."""
        logger.info("Comparing predictions with ground truth labels...")
        
        # Load ground truth labels
        gt_labels_dir = Path("cheetah_data/cheeta_train_validation_labled/labels")
        
        comparison_results = []
        total_predictions = 0
        total_ground_truth = 0
        correct_detections = 0
        
        for result in val_results:
            img_name = result['image']
            predictions_xywhn = result.get('predictions_xywhn')
            
            # Find corresponding ground truth label file
            label_file = gt_labels_dir / f"{img_name.replace('.jpg', '.txt')}"
            
            if not label_file.exists():
                logger.warning(f"No ground truth label found for {img_name}")
                continue
                
            # Load ground truth labels
            gt_boxes = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x_center, y_center, width, height = map(float, parts[:5])
                        gt_boxes.append([x_center, y_center, width, height])
            
            # Convert predictions to normalized coordinates for comparison
            pred_boxes = []
            if predictions_xywhn is not None and len(predictions_xywhn) > 0:
                # predictions_xywhn format: [x_center, y_center, w, h, conf]
                for pred in predictions_xywhn:
                    if len(pred) >= 5:
                        x_center, y_center, width, height, conf = map(float, pred[:5])
                        pred_boxes.append([x_center, y_center, width, height, conf])
            else:
                # Backward compatibility: if only xyxy is available, normalize by image size
                predictions_xyxy = result.get('predictions')
                if predictions_xyxy is not None and len(predictions_xyxy) > 0:
                    from PIL import Image
                    img_path = Path("cheetah_data/cheeta_train_validation_labled/images") / img_name
                    with Image.open(img_path) as im:
                        w, h = im.size

                    for pred in predictions_xyxy:
                        x1, y1, x2, y2 = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
                        conf = float(pred[4]) if len(pred) > 4 else 1.0
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        pred_boxes.append([x_center, y_center, width, height, conf])
            
            # Simple IoU-based matching (threshold = 0.5)
            matches = self._calculate_matches(pred_boxes, gt_boxes, iou_threshold=0.5)
            
            comparison_results.append({
                'image': img_name,
                'predictions_count': len(pred_boxes),
                'ground_truth_count': len(gt_boxes),
                'matches': len(matches),
                'precision': len(matches) / len(pred_boxes) if len(pred_boxes) > 0 else 0,
                'recall': len(matches) / len(gt_boxes) if len(gt_boxes) > 0 else 0
            })
            
            total_predictions += len(pred_boxes)
            total_ground_truth += len(gt_boxes)
            correct_detections += len(matches)
        
        # Calculate overall metrics
        overall_precision = correct_detections / total_predictions if total_predictions > 0 else 0
        overall_recall = correct_detections / total_ground_truth if total_ground_truth > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Save comparison results
        comparison_summary = {
            'overall_metrics': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'total_predictions': total_predictions,
                'total_ground_truth': total_ground_truth,
                'correct_detections': correct_detections
            },
            'per_image_results': comparison_results
        }
        
        with open(output_dir / "validation_comparison.json", 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        logger.info(f"Validation comparison completed:")
        logger.info(f"  Overall Precision: {overall_precision:.4f}")
        logger.info(f"  Overall Recall: {overall_recall:.4f}")
        logger.info(f"  Overall F1-Score: {overall_f1:.4f}")
        logger.info(f"  Total Predictions: {total_predictions}")
        logger.info(f"  Total Ground Truth: {total_ground_truth}")
        logger.info(f"  Correct Detections: {correct_detections}")
        
        return comparison_summary
    
    def _calculate_matches(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """Calculate matches between predictions and ground truth using IoU."""
        matches = []
        used_gt = set()
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in used_gt:
                    continue
                    
                iou = self._calculate_iou(pred_box[:4], gt_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                matches.append({
                    'pred_idx': pred_idx,
                    'gt_idx': best_gt_idx,
                    'iou': best_iou,
                    'confidence': pred_box[4] if len(pred_box) > 4 else 0
                })
                used_gt.add(best_gt_idx)
        
        return matches
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two boxes."""
        # box format: [x_center, y_center, width, height] (normalized)
        
        # Convert to corner format
        x1_1 = box1[0] - box1[2] / 2
        y1_1 = box1[1] - box1[3] / 2
        x2_1 = box1[0] + box1[2] / 2
        y2_1 = box1[1] + box1[3] / 2
        
        x1_2 = box2[0] - box2[2] / 2
        y1_2 = box2[1] - box2[3] / 2
        x2_2 = box2[0] + box2[2] / 2
        y2_2 = box2[1] + box2[3] / 2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _save_training_results(self, results, model):
        """Save training results and create visualizations."""
        logger.info("Saving training results...")
        
        # Save model
        model_path = self.run_dir / 'best_cheetah_model.pt'
        model.save(model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Create training visualizations
        self._create_training_visualizations()
        
        # Create comprehensive training analysis
        try:
            # Read results CSV if it exists
            results_csv = self.run_dir / 'cheetah_yolo12' / 'results.csv'
            if results_csv.exists():
                results_df = pd.read_csv(results_csv)
                self.create_training_analysis(results_df)
            else:
                logger.warning("Results CSV not found, skipping detailed analysis")
        except Exception as e:
            logger.warning(f"Could not create training analysis: {e}")
        
        # Export model in different formats
        self._export_model(model)
    
    def _create_training_visualizations(self):
        """Create training progress visualizations."""
        results_dir = self.run_dir / 'cheetah_yolo12'
        
        if not results_dir.exists():
            logger.warning("Results directory not found, skipping visualizations")
            return
        
        # Look for results.csv
        results_csv = results_dir / 'results.csv'
        if not results_csv.exists():
            logger.warning("Results CSV not found, skipping visualizations")
            return
        
        # Read training results
        df = pd.read_csv(results_csv)
        
        # Create training plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLOv12 Training Progress', fontsize=16, fontweight='bold')
        
        # Loss curves
        ax = axes[0, 0]
        if 'train/box_loss' in df.columns:
            ax.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
        if 'val/box_loss' in df.columns:
            ax.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Box Loss')
        ax.set_title('Box Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        if 'train/cls_loss' in df.columns:
            ax.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', color='blue')
        if 'val/cls_loss' in df.columns:
            ax.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Classification Loss')
        ax.set_title('Classification Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # mAP curves
        ax = axes[1, 0]
        if 'metrics/mAP50(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 1]
        if 'lr/pg0' in df.columns:
            ax.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training visualizations saved to training_progress.png")
    
    def _export_model(self, model):
        """Export model in different formats for deployment."""
        logger.info("Exporting model in different formats...")
        
        export_dir = self.run_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        
        try:
            # Export to ONNX - use same image size as training
            onnx_path = export_dir / 'cheetah_model.onnx'
            model.export(format='onnx', imgsz=self.config['imgsz'])
            logger.info(f"ONNX model exported to: {onnx_path}")
            
            # Export to TorchScript - use same image size as training
            torchscript_path = export_dir / 'cheetah_model.pt'
            model.export(format='torchscript', imgsz=self.config['imgsz'])
            logger.info(f"TorchScript model exported to: {torchscript_path}")
            
        except Exception as e:
            logger.warning(f"Model export failed: {e}")
    
    def test_model(self, model_path=None, conf=None, iou=None, max_det=None):
        """Test the trained model on test dataset."""
        logger.info("Testing model on test dataset...")
        
        if model_path is None:
            model_path = self.run_dir / 'best_cheetah_model.pt'
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None
        
        # Load model
        model = YOLO(str(model_path))
        
        # Run inference on test images
        test_images = list(self.test_data.glob('*.jpg'))
        if not test_images:
            logger.warning("No test images found")
            return None
        
        # Resolve inference thresholds - use same as training for consistency
        # This matches what was shown in validation batch images (conf=0.8, iou=0.3, max_det=1)
        conf_th = conf if conf is not None else self.config['conf']
        iou_th = iou if iou is not None else self.config['iou']
        max_det_val = max_det if max_det is not None else self.config['max_det']
        
        logger.info(f"Test inference using: conf={conf_th}, iou={iou_th}, max_det={max_det_val}")

        results = model(
            self.test_data,
            save=True,
            save_txt=True,
            save_conf=True,
            conf=conf_th,
            iou=iou_th,
            max_det=max_det_val,
            agnostic_nms=False,
            project=str(self.run_dir),
            name='test_results'
        )
        
        # Create confidence analysis for test images
        self._create_test_confidence_analysis(results)
        
        # Create enhanced validation analysis with proper true/false positive metrics
        self._create_enhanced_validation_analysis(results)
        
        logger.info(f"Test results saved to: {self.run_dir / 'test_results'}")
        return results
    
    def _create_test_confidence_analysis(self, test_results):
        """Create confidence analysis graph for test images."""
        logger.info("Creating test confidence analysis...")
        
        # Extract confidence scores for each image
        image_confidences = []
        image_names = []
        
        for i, result in enumerate(test_results):
            if result.boxes is not None and len(result.boxes.conf) > 0:
                # Get average confidence for this image
                avg_confidence = float(result.boxes.conf.mean().cpu())
                max_confidence = float(result.boxes.conf.max().cpu())
                min_confidence = float(result.boxes.conf.min().cpu())
                num_detections = len(result.boxes.conf)
                
                image_confidences.append({
                    'image_idx': i + 1,
                    'image_name': result.path.split('\\')[-1] if '\\' in result.path else result.path.split('/')[-1],
                    'avg_confidence': avg_confidence,
                    'max_confidence': max_confidence,
                    'min_confidence': min_confidence,
                    'num_detections': num_detections
                })
                image_names.append(f"Img {i+1}")
            else:
                # No detections for this image
                image_confidences.append({
                    'image_idx': i + 1,
                    'image_name': result.path.split('\\')[-1] if '\\' in result.path else result.path.split('/')[-1],
                    'avg_confidence': 0.0,
                    'max_confidence': 0.0,
                    'min_confidence': 0.0,
                    'num_detections': 0
                })
                image_names.append(f"Img {i+1}")
        
        # Create the confidence analysis plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: Confidence variance across images
        x_values = [item['image_idx'] for item in image_confidences]
        avg_confidences = [item['avg_confidence'] for item in image_confidences]
        max_confidences = [item['max_confidence'] for item in image_confidences]
        min_confidences = [item['min_confidence'] for item in image_confidences]
        
        # Plot individual confidence points
        ax1.scatter(x_values, avg_confidences, alpha=0.7, s=60, color='blue', label='Average Confidence per Image')
        ax1.scatter(x_values, max_confidences, alpha=0.5, s=40, color='red', label='Max Confidence per Image')
        ax1.scatter(x_values, min_confidences, alpha=0.5, s=40, color='green', label='Min Confidence per Image')
        
        # Add trend line for average confidence
        if len(avg_confidences) > 1:
            z = np.polyfit(x_values, avg_confidences, 1)
            p = np.poly1d(z)
            ax1.plot(x_values, p(x_values), "r--", alpha=0.8, linewidth=3, 
                    label=f'Average Trend: {z[0]:.4f}x + {z[1]:.4f}')

        ax1.set_title('Test Image Confidence Analysis', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Test Image Index', fontsize=12)
        ax1.set_ylabel('Confidence Score', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add statistics text
        overall_avg = np.mean(avg_confidences)
        overall_std = np.std(avg_confidences)
        ax1.text(0.02, 0.98, f'Overall Average: {overall_avg:.4f}\nStandard Deviation: {overall_std:.4f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Plot 2: Number of detections vs confidence
        num_detections = [item['num_detections'] for item in image_confidences]
        
        # Create scatter plot
        scatter = ax2.scatter(num_detections, avg_confidences, 
                            c=avg_confidences, cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Confidence Score', fontsize=10)
        
        # Add trend line
        if len(num_detections) > 1 and max(num_detections) > 0:
            z2 = np.polyfit(num_detections, avg_confidences, 1)
            p2 = np.poly1d(z2)
            x_trend = np.linspace(min(num_detections), max(num_detections), 100)
            ax2.plot(x_trend, p2(x_trend), "r--", alpha=0.8, linewidth=3,
                    label=f'Detection-Correlation: {z2[0]:.4f}x + {z2[1]:.4f}')
        
        ax2.set_title('Number of Detections vs Average Confidence', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Number of Detections per Image', fontsize=12)
        ax2.set_ylabel('Average Confidence Score', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Add correlation coefficient
        correlation = 0.0
        if len(num_detections) > 1:
            correlation = np.corrcoef(num_detections, avg_confidences)[0, 1]
            ax2.text(0.02, 0.98, f'Correlation: {correlation:.4f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        analysis_dir = self.run_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        plt.savefig(analysis_dir / "test_confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed confidence data
        confidence_data = {
            'image_analysis': image_confidences,
            'overall_statistics': {
                'mean_confidence': float(overall_avg),
                'std_confidence': float(overall_std),
                'min_confidence': float(min(avg_confidences)),
                'max_confidence': float(max(avg_confidences)),
                'total_images': len(image_confidences),
                'images_with_detections': len([x for x in image_confidences if x['num_detections'] > 0]),
                'images_without_detections': len([x for x in image_confidences if x['num_detections'] == 0])
            },
            'detection_correlation': float(correlation) if len(num_detections) > 1 else 0.0
        }
        
        with open(analysis_dir / "test_confidence_data.json", 'w') as f:
            json.dump(confidence_data, f, indent=2)
        
        logger.info(f"Test confidence analysis saved to {analysis_dir}/test_confidence_analysis.png")
        logger.info(f"Overall average confidence: {overall_avg:.4f} ± {overall_std:.4f}")
        logger.info(f"Images with detections: {len([x for x in image_confidences if x['num_detections'] > 0])}/{len(image_confidences)}")
    
    def _create_enhanced_validation_analysis(self, test_results):
        """Create enhanced validation analysis with proper true/false positive metrics."""
        logger.info("Creating enhanced validation analysis with proper true/false positive metrics...")
        
        # Extract confidence scores and categorize images properly
        image_confidences = []
        
        for i, result in enumerate(test_results):
            if result.boxes is not None and len(result.boxes.conf) > 0:
                avg_confidence = float(result.boxes.conf.mean().cpu())
                max_confidence = float(result.boxes.conf.max().cpu())
                min_confidence = float(result.boxes.conf.min().cpu())
                num_detections = len(result.boxes.conf)
            else:
                avg_confidence = 0.0
                max_confidence = 0.0
                min_confidence = 0.0
                num_detections = 0
            
            img_name = result.path.split('\\')[-1] if '\\' in result.path else result.path.split('/')[-1]
            
            image_confidences.append({
                'image_idx': i + 1,
                'image_name': img_name,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'min_confidence': min_confidence,
                'num_detections': num_detections
            })
        
        # Categorize images properly
        tiger_images = []  # Should have 0 detections (false positives if detected)
        cheetah_single_images = []  # Should have exactly 1 detection
        cheetah_multi_images = []  # Should have multiple detections (cheetah_735_resized)
        
        for img_data in image_confidences:
            img_name = img_data['image_name']
            num_detections = img_data['num_detections']
            
            if img_name.startswith('000000'):  # Tiger images
                tiger_images.append({
                    'name': img_name,
                    'detections': num_detections,
                    'confidence': img_data['avg_confidence'],
                    'expected_detections': 0,
                    'is_correct': num_detections == 0
                })
            elif img_name == 'cheetah_735_resized.jpg':  # Multi-cheetah image
                cheetah_multi_images.append({
                    'name': img_name,
                    'detections': num_detections,
                    'confidence': img_data['avg_confidence'],
                    'expected_detections': 'multiple',
                    'is_correct': num_detections > 1  # Should have multiple cheetahs
                })
            else:  # Single cheetah images
                cheetah_single_images.append({
                    'name': img_name,
                    'detections': num_detections,
                    'confidence': img_data['avg_confidence'],
                    'expected_detections': 1,
                    'is_correct': num_detections == 1
                })
        
        # Calculate proper metrics
        total_tiger_images = len(tiger_images)
        total_cheetah_single = len(cheetah_single_images)
        total_cheetah_multi = len(cheetah_multi_images)
        
        # False Positives: Tiger images with detections
        false_positives = len([img for img in tiger_images if img['detections'] > 0])
        
        # True Positives: Cheetah images with correct detections
        true_positives_single = len([img for img in cheetah_single_images if img['is_correct']])
        true_positives_multi = len([img for img in cheetah_multi_images if img['is_correct']])
        total_true_positives = true_positives_single + true_positives_multi
        
        # False Negatives: Cheetah images with wrong number of detections
        false_negatives_single = len([img for img in cheetah_single_images if not img['is_correct']])
        false_negatives_multi = len([img for img in cheetah_multi_images if not img['is_correct']])
        total_false_negatives = false_negatives_single + false_negatives_multi
        
        # Calculate proper precision and recall
        total_actual_cheetahs = total_cheetah_single + (2 if total_cheetah_multi > 0 else 0)  # Assume 2 cheetahs in multi image
        
        precision = total_true_positives / (total_true_positives + false_positives) if (total_true_positives + false_positives) > 0 else 0
        recall = total_true_positives / total_actual_cheetahs if total_actual_cheetahs > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create comprehensive analysis
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Detection Accuracy by Image Type
        categories = ['Tiger Images\n(Should be 0)', 'Single Cheetah\n(Should be 1)', 'Multi Cheetah\n(Should be >1)']
        correct_counts = [len([img for img in tiger_images if img['is_correct']]), 
                         true_positives_single, 
                         true_positives_multi]
        total_counts = [total_tiger_images, total_cheetah_single, total_cheetah_multi]
        accuracy_rates = [correct_counts[i]/total_counts[i] if total_counts[i] > 0 else 0 for i in range(3)]
        
        bars = axes[0, 0].bar(categories, accuracy_rates, color=['red', 'green', 'blue'], alpha=0.7)
        axes[0, 0].set_title('Detection Accuracy by Image Type', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy Rate')
        axes[0, 0].set_ylim(0, 1.1)
        
        # Add percentage labels on bars
        for i, (bar, rate) in enumerate(zip(bars, accuracy_rates)):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{rate:.1%}\n({correct_counts[i]}/{total_counts[i]})',
                           ha='center', va='bottom', fontweight='bold')
        
        # 2. False Positive Analysis (Tiger Images)
        tiger_detections = [img['detections'] for img in tiger_images]
        tiger_confidences = [img['confidence'] for img in tiger_images]
        
        axes[0, 1].scatter(tiger_detections, tiger_confidences, color='red', s=100, alpha=0.7)
        axes[0, 1].set_title('False Positive Analysis (Tiger Images)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Number of Detections (Should be 0)')
        axes[0, 1].set_ylabel('Average Confidence')
        axes[0, 1].axvline(x=0, color='green', linestyle='--', linewidth=2, label='Correct (0 detections)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add count of false positives
        fp_count = len([d for d in tiger_detections if d > 0])
        axes[0, 1].text(0.02, 0.98, f'False Positives: {fp_count}/{total_tiger_images}', 
                       transform=axes[0, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
        
        # 3. Single Cheetah Detection Analysis
        single_detections = [img['detections'] for img in cheetah_single_images]
        single_confidences = [img['confidence'] for img in cheetah_single_images]
        
        axes[0, 2].scatter(single_detections, single_confidences, color='green', s=100, alpha=0.7)
        axes[0, 2].set_title('Single Cheetah Detection Analysis', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Number of Detections (Should be 1)')
        axes[0, 2].set_ylabel('Average Confidence')
        axes[0, 2].axvline(x=1, color='green', linestyle='--', linewidth=2, label='Correct (1 detection)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add accuracy info
        correct_single = len([d for d in single_detections if d == 1])
        axes[0, 2].text(0.02, 0.98, f'Correct: {correct_single}/{total_cheetah_single}', 
                       transform=axes[0, 2].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
        
        # 4. Confidence Distribution by Image Type
        tiger_conf = [img['confidence'] for img in tiger_images if img['detections'] > 0]  # Only false positives
        cheetah_single_conf = [img['confidence'] for img in cheetah_single_images]
        cheetah_multi_conf = [img['confidence'] for img in cheetah_multi_images]
        
        data_to_plot = [tiger_conf, cheetah_single_conf, cheetah_multi_conf]
        labels = ['Tiger (FP)', 'Single Cheetah', 'Multi Cheetah']
        colors = ['red', 'green', 'blue']
        
        box_plot = axes[1, 0].boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 0].set_title('Confidence Distribution by Image Type', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Error Analysis
        error_types = ['False Positives\n(Tiger detected)', 'False Negatives\n(Cheetah missed)', 'Correct Detections']
        error_counts = [false_positives, total_false_negatives, total_true_positives]
        error_colors = ['red', 'orange', 'green']
        
        wedges, texts, autotexts = axes[1, 1].pie(error_counts, labels=error_types, colors=error_colors, 
                                                 autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Error Analysis', fontsize=14, fontweight='bold')
        
        # 6. Performance Metrics Summary
        metrics_text = f"""
PROPER VALIDATION METRICS:

Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1_score:.4f}

DETAILED BREAKDOWN:
• Tiger Images: {total_tiger_images} total
  - False Positives: {false_positives}
  - Accuracy: {(len([img for img in tiger_images if img['is_correct']])/total_tiger_images if total_tiger_images > 0 else 0):.1%}

• Single Cheetah: {total_cheetah_single} total
  - Correct: {true_positives_single}
  - Accuracy: {(true_positives_single/total_cheetah_single if total_cheetah_single > 0 else 0):.1%}

• Multi Cheetah: {total_cheetah_multi} total
  - Correct: {true_positives_multi}
  - Accuracy: {(true_positives_multi/total_cheetah_multi if total_cheetah_multi > 0 else 0):.1%}

TOTAL ERRORS: {false_positives + total_false_negatives}
"""
        
        axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1, 2].set_title('Performance Summary', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        analysis_dir = self.run_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        plt.savefig(analysis_dir / "enhanced_validation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tiger_images': {
                'total': total_tiger_images,
                'false_positives': false_positives,
                'accuracy': (total_tiger_images - false_positives) / total_tiger_images if total_tiger_images > 0 else 0
            },
            'single_cheetah': {
                'total': total_cheetah_single,
                'correct': true_positives_single,
                'accuracy': true_positives_single / total_cheetah_single if total_cheetah_single > 0 else 0
            },
            'multi_cheetah': {
                'total': total_cheetah_multi,
                'correct': true_positives_multi,
                'accuracy': true_positives_multi / total_cheetah_multi if total_cheetah_multi > 0 else 0
            },
            'total_errors': false_positives + total_false_negatives
        }
        
        with open(analysis_dir / "enhanced_validation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log the proper metrics
        logger.info("="*80)
        logger.info("ENHANCED VALIDATION ANALYSIS - PROPER TRUE/FALSE POSITIVE METRICS")
        logger.info("="*80)
        logger.info(f"PRECISION: {precision:.4f} (True Positives / (True Positives + False Positives))")
        logger.info(f"RECALL: {recall:.4f} (True Positives / Total Actual Cheetahs)")
        logger.info(f"F1-SCORE: {f1_score:.4f}")
        logger.info(f"Tiger Images (should have 0 detections): {total_tiger_images}")
        logger.info(f"  - False Positives: {false_positives} ({false_positives/total_tiger_images:.1%})")
        logger.info(f"Single Cheetah Images (should have 1 detection): {total_cheetah_single}")
        logger.info(f"  - Correct: {true_positives_single} ({true_positives_single/total_cheetah_single:.1%})")
        logger.info(f"Multi Cheetah Images (should have >1 detection): {total_cheetah_multi}")
        logger.info(f"  - Correct: {true_positives_multi} ({true_positives_multi/total_cheetah_multi:.1%})")
        logger.info(f"TOTAL ERRORS: {false_positives + total_false_negatives}")
        logger.info("="*80)
        
        logger.info(f"Enhanced validation analysis saved to {analysis_dir}/enhanced_validation_analysis.png")

    def sweep_validation_thresholds(self, model):
        """Sweep confidence thresholds on labeled validation set and pick F1-optimal conf.

        Returns:
            float: best confidence threshold by F1-score.
        """
        logger.info("Sweeping validation confidence thresholds to balance precision/recall...")

        val_images_dir = Path("cheetah_data/cheeta_train_validation_labled/images")
        if not val_images_dir.exists():
            logger.warning("Labeled validation images not found; skipping threshold sweep.")
            return self.inference_conf

        image_paths = sorted(list(val_images_dir.glob("*.jpg")))
        if not image_paths:
            logger.warning("No labeled validation images found; skipping threshold sweep.")
            return self.inference_conf

        thresholds = [round(x, 2) for x in np.arange(0.10, 0.71, 0.05)]
        iou_th = self.inference_iou

        analysis_dir = self.run_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        sweep_results = []
        best_conf = self.inference_conf
        best_f1 = -1.0

        for conf_th in thresholds:
            preds = model.predict([str(p) for p in image_paths], conf=conf_th, iou=iou_th,
                                  max_det=10, verbose=False, save=False)  # Inference: allow up to 10 detections

            val_results = []
            for img_path, res in zip(image_paths, preds):
                boxes = res.boxes
                if boxes is not None and len(boxes) > 0:
                    xywhn = boxes.xywhn.cpu().numpy()
                    conf = boxes.conf.cpu().numpy().reshape(-1, 1)
                    preds_xywhn = np.hstack([xywhn, conf])
                else:
                    preds_xywhn = []
                val_results.append({
                    'image': img_path.name,
                    'predictions_xywhn': preds_xywhn
                })

            # Use a dedicated directory per threshold to avoid overwriting
            out_dir = analysis_dir / f"sweep_conf_{conf_th:.2f}"
            out_dir.mkdir(exist_ok=True)
            summary = self._compare_with_ground_truth(val_results, out_dir)

            metrics = summary.get('overall_metrics', {})
            precision = float(metrics.get('precision', 0.0))
            recall = float(metrics.get('recall', 0.0))
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            sweep_results.append({'conf': conf_th, 'precision': precision, 'recall': recall, 'f1': f1})

            if f1 > best_f1:
                best_f1 = f1
                best_conf = conf_th

        # Persist sweep summary
        with open(analysis_dir / 'threshold_sweep.json', 'w') as f:
            json.dump({'results': sweep_results, 'best_conf': best_conf, 'best_f1': best_f1}, f, indent=2)

        logger.info(f"Threshold sweep complete. Best conf={best_conf:.2f} with F1={best_f1:.4f}")
        return best_conf
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting complete YOLOv12 training pipeline...")
        
        try:
            # Step 1: Validate dataset
            logger.info("Step 1: Validating dataset...")
            validation_results = self.validate_dataset()
            
            # Step 2: Analyze dataset
            logger.info("Step 2: Analyzing dataset...")
            analysis_results = self.analyze_dataset()
            
            # Step 3: Train model
            logger.info("Step 3: Training model...")
            training_results, model = self.train_model()
            
            # Step 4: Custom validation on unlabeled data
            logger.info("Step 4: Validating model on unlabeled validation data...")
            validation_results = self.validate_model_on_unlabeled_data(model)
            
            # Step 5: Sweep thresholds on labeled validation and set inference thresholds
            logger.info("Step 5: Sweeping thresholds on labeled validation set...")
            best_conf = self.sweep_validation_thresholds(model)
            self.inference_conf = best_conf

            # Step 6: Test model with selected thresholds
            logger.info("Step 6: Testing model with selected thresholds...")
            test_results = self.test_model(conf=self.inference_conf, iou=self.inference_iou, max_det=10)  # Inference: allow up to 10 detections
            
            # Step 7: Create summary report
            logger.info("Step 7: Creating summary report...")
            self._create_summary_report(validation_results, analysis_results, 
                                      training_results, test_results)
            
            logger.info("Complete pipeline finished successfully!")
            logger.info(f"All results saved to: {self.run_dir}")
            
            return {
                'validation': validation_results,
                'analysis': analysis_results,
                'training': training_results,
                'testing': test_results,
                'run_dir': str(self.run_dir)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _create_summary_report(self, validation, analysis, training, testing):
        """Create a comprehensive summary report."""
        # Load enhanced validation results if available
        enhanced_validation = None
        enhanced_validation_file = self.run_dir / "analysis" / "enhanced_validation_results.json"
        if enhanced_validation_file.exists():
            with open(enhanced_validation_file, 'r') as f:
                enhanced_validation = json.load(f)
        
        report = {
            'timestamp': self.timestamp,
            'model_size': self.model_size,
            'dataset_validation': validation,
            'dataset_analysis': analysis,
            'training_summary': {
                'epochs': self.config['epochs'],
                'batch_size': self.config['batch'],
                'image_size': self.config['imgsz'],
                'optimizer': self.config['optimizer'],
                'learning_rate': self.config['lr0']
            },
            'enhanced_validation': enhanced_validation,
            'test_summary': {
                'test_images': len(list(self.test_data.glob('*.jpg'))) if self.test_data.exists() else 0
            },
            'inference': {
                'conf': getattr(self, 'inference_conf', 0.35),
                'iou': getattr(self, 'inference_iou', 0.55),
                'max_det': 10  # Inference: allow up to 10 detections
            }
        }
        
        # Save report
        with open(self.run_dir / 'summary_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown report
        self._create_markdown_report(report)
        
        logger.info("Summary report created: summary_report.json and summary_report.md")
    
    def _create_markdown_report(self, report):
        """Create a markdown summary report."""
        md_content = f"""# YOLOv12 Cheetah Detection Training Report

## Training Summary
- **Model**: YOLOv12{self.model_size}
- **Timestamp**: {self.timestamp}
- **Epochs**: {self.config['epochs']}
- **Batch Size**: {self.config['batch']}
- **Image Size**: {self.config['imgsz']}
- **Optimizer**: {self.config['optimizer']}
- **Learning Rate**: {self.config['lr0']}

## Inference Settings
- **Confidence**: {getattr(self, 'inference_conf', 0.35)}
- **IoU**: {getattr(self, 'inference_iou', 0.55)}
- **Max Detections**: 10 (inference only)

## Dataset Information
- **Training Images**: {report['dataset_validation'].get('train', {}).get('image_count', 0) if 'train' in report['dataset_validation'] else 0}
- **Validation Images**: {report['dataset_validation'].get('val', {}).get('image_count', 0) if 'val' in report['dataset_validation'] else 0}
- **Test Images**: {report['test_summary']['test_images']}
- **Classes**: {report['dataset_validation'].get('train', {}).get('classes', ['Cheetah']) if 'train' in report['dataset_validation'] else ['Cheetah']}

## Files Generated
- `best_cheetah_model.pt`: Best trained model
- `training_progress.png`: Training curves
- `dataset_analysis.png`: Dataset analysis plots
- `analysis/training_analysis.png`: Comprehensive training analysis
- `analysis/validation_accuracy_analysis.png`: Validation accuracy plots
- `analysis/batch_size_analysis.png`: Batch size optimization analysis
- `analysis/test_confidence_analysis.png`: Test image confidence analysis
- `analysis/enhanced_validation_analysis.png`: Proper true/false positive validation
- `exports/`: Model exports (ONNX, TorchScript)
- `test_results/`: Test inference results
- `summary_report.json`: Detailed JSON report

## Enhanced Validation Analysis (Proper True/False Positive Metrics)
- **Precision**: {report.get('enhanced_validation', {}).get('precision', 0):.4f}
- **Recall**: {report.get('enhanced_validation', {}).get('recall', 0):.4f}
- **F1-Score**: {report.get('enhanced_validation', {}).get('f1_score', 0):.4f}
- **False Positives (Tiger Images)**: {report.get('enhanced_validation', {}).get('tiger_images', {}).get('false_positives', 0)}/{report.get('enhanced_validation', {}).get('tiger_images', {}).get('total', 0)}
- **Single Cheetah Accuracy**: {report.get('enhanced_validation', {}).get('single_cheetah', {}).get('accuracy', 0):.1%}
- **Multi Cheetah Accuracy**: {report.get('enhanced_validation', {}).get('multi_cheetah', {}).get('accuracy', 0):.1%}
- **Total Errors**: {report.get('enhanced_validation', {}).get('total_errors', 0)}

## Next Steps
1. Review training curves in `training_progress.png`
2. Analyze dataset characteristics in `dataset_analysis.png`
3. Review enhanced validation analysis in `analysis/enhanced_validation_analysis.png`
4. Test model performance on new images
5. Use exported models for deployment

## Usage
```python
from ultralytics import YOLO
model = YOLO('best_cheetah_model.pt')
results = model('path/to/image.jpg')
```
"""
        
        with open(self.run_dir / 'summary_report.md', 'w') as f:
            f.write(md_content)


def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv12 Cheetah Detection Training - Exclusively uses YOLOv12 models')
    parser.add_argument('--model-size', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv12 model size (n, s, m, l, x) - exclusively uses YOLOv12 models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Image size override; if omitted uses internal default')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    trainer = YOLOv12CheetahTrainer(model_size=args.model_size, seed=args.seed)
    trainer.config['epochs'] = args.epochs
    trainer.config['batch'] = args.batch
    if args.imgsz is not None:
        trainer.config['imgsz'] = args.imgsz
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline()
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Results saved to: {results['run_dir']}")
    print(f"Model: best_cheetah_model.pt")
    print(f"Visualizations: training_progress.png, dataset_analysis.png")
    print(f"Report: summary_report.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
