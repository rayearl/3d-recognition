# 3D Face Recognition using PointNet and ArcFace

This repository contains an implementation of a 3D face recognition system using PointNet and ArcFace. The system processes 3D point cloud data of faces for accurate recognition in various applications such as security, access control, and biometric authentication.



## Features

- 3D face recognition using point cloud data
- Implementation of PointNet architecture for point cloud processing
- ArcFace loss function for enhanced recognition accuracy
- Data preprocessing and alignment pipeline
- Visualization tools for 3D point cloud data
- Training with TensorBoard integration for monitoring
- Evaluation metrics for model performance

## Getting Started

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Data Preparation

Before training, you need to align the raw 3D face data:

```bash
python align-training-dataset.py --input_dir /path/to/raw_data --output_dir /path/to/aligned_data
```

### Training

To train the model:
```bash
python train.py --data_dir /path/to/aligned_data --batch_size 32 --epochs 100 --arc_margin 0.5 --arc_scale 30.0
```
