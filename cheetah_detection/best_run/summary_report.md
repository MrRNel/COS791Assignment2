# YOLOv12 Cheetah Detection Training Report

## Training Summary
- **Model**: YOLOv12n
- **Timestamp**: 20251026_150504
- **Epochs**: 100
- **Batch Size**: 95
- **Image Size**: 400
- **Optimizer**: AdamW
- **Learning Rate**: 0.001

## Inference Settings
- **Confidence**: 0.7
- **IoU**: 0.3
- **Max Detections**: 10 (inference only)

## Dataset Information
- **Training Images**: 0
- **Validation Images**: 0
- **Test Images**: 46
- **Classes**: ['Cheetah']

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
- **Precision**: 1.0000
- **Recall**: 0.8974
- **F1-Score**: 0.9459
- **False Positives (Tiger Images)**: 0/8
- **Single Cheetah Accuracy**: 94.6%
- **Multi Cheetah Accuracy**: 0.0%
- **Total Errors**: 3

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
