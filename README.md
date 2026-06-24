# Satellite Pose Estimation (SPEED Challenge)

3D pose estimation for the ESA [Kelvins Satellite Pose Estimation Challenge](https://kelvins.esa.int/satellite-pose-estimation-challenge/) using the SPEED dataset. This repository explores CNN-based regression of satellite orientation (quaternions) and position vectors from monocular 2D images.

## Repository Structure

```
PoseEstimation/
├── src/                    # Reusable Python modules
│   ├── data/               # Dataset loaders, generators, augmentations
│   ├── models/             # Model architecture definitions
│   │   ├── resnet34.py
│   │   ├── inception_v4.py, InceptionResnetv2.py
│   │   └── hourglass.py, hg_blocks.py, HourglassPose.py
│   ├── losses/             # Custom loss functions
│   ├── layers/             # Custom Keras layers
│   │   ├── uncertainty_layer.py
│   │   └── AdamAccumulateDutil.py
│   └── utils/              # Utilities (submissions, schedulers, metrics)
│       ├── submission.py
│       ├── rotation.py
│       └── cyclical_learning_rate.py
│
├── experiments/            # Training scripts organized by architecture
│   ├── baseline/           # Starter kit examples (Keras + PyTorch)
│   ├── inception/          # Inceptionv3 experiments
│   ├── densenet/           # DenseNet121 experiments
│   ├── hourglass/          # Hourglass, ResNeXt, uncertainty testers
│   ├── siamese/            # Siamese network experiments
│   ├── attention/          # Attention Network (CBAM) experiments
│   ├── hybrid/             # LSTM, Faster-RCNN hybrid approaches
│   ├── production_a/       # Production-ready training pipeline
│   └── 2020_investigations/# Later experiments (2020)
│
├── scripts/                # Standalone utility scripts
│   ├── measurebrightness.py
│   ├── PlotPredictions.py
│   ├── main_lr_finder.py
│   └── kfold_cv.py
│
├── data/                   # SPEED dataset
│   └── speed/              # Preprocessed images and annotations
│
├── results/                # All experiment outputs
│   ├── submissions/        # Competition submission CSV files
│   ├── models/             # Saved weights (.h5, .json, .obj)
│   ├── visualizations/     # HTML plots, PNG loss curves
│   ├── logs/               # TensorBoard training logs
│   ├── best_model/         # Best performing model files
│   └── EXPERIMENT_RESULTS.md  # Consolidated results tracker
│
├── notebooks/              # Jupyter notebooks
│   └── visualize_pose.ipynb
│
├── papers/                 # Research papers
├── third_party/            # Third-party code (git submodules or standalone)
├── archive/                # Backups and zip/rar archives
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md
```

## Architectures Explored

- **ResNet** (34, 50, 101) — baseline CNNs with residual connections
- **Inceptionv3/v4 / Inception-ResNet-v2** — most heavily experimented with; best performer
- **DenseNet121** — with/without regularization, uncertainty prediction, LSTM heads
- **EfficientNet** (B3, B4) — 2020 experiments
- **Hourglass** — stacked hourglass networks for pose estimation
- **ResNeXt** — aggregated residual transformations
- **CBAM** — Convolutional Block Attention Module with Inceptionv3
- **Siamese** — siamese network architecture
- **Faster R-CNN** — object detection + pose regression hybrid

## Key Techniques

- **Loss functions**: MSE, geodesic loss, quaternion loss, uncertainty-weighted losses, auxiliary regression losses
- **Augmentation**: rotation-only, rotation+translation, illumination
- **Regularization**: L2 weight decay, batch normalization, dropout
- **Optimization**: Adam with gradient accumulation, cyclical LR, ReduceLROnPlateau
- **Output heads**: dual-head (quaternion + position), with learned uncertainty

## Quick Start

```bash
pip install -r requirements.txt
# Download SPEED dataset and place in data/speed/
# Then run an experiment:
python experiments/baseline/keras_example.py --dataset data/speed --epochs 100 --batch 32
```

## Results

Best Inceptionv3 model achieved **2.1865 LB score**. See `results/EXPERIMENT_RESULTS.md` for full comparison across architectures and hyperparameters.
