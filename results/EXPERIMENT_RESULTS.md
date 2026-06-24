# Results Tracker — Satellite Pose Estimation (SPEED Challenge)

## Competition Leaderboard Scores
| Date | User | Model | Resolution | Augmentation | Loss | Leaderboard Score |
|------|------|-------|-----------|-------------|------|-----------------|
| 18-05-2019 | - | Baseline | - | - | - | 2.8016 (real) / 2.7497 (best) |
| 18-05-2019 | - | - | - | - | - | 2.7957 |
| 18-05-2019 | - | - | - | - | - | 2.9582 / 2.3706 |
| 29-05-2019 | GabrielA | ID001 | 224 | None | MSE (1:1 q:r) | 2.8323 / 2.2743 |
| 30-05-2019 | GabrielA | - | 224 | None | q-loss + norm penalty | 2.4483 |
| 30-05-2019 | GabrielA | - | 224 | Angle aug | MSE | 2.3151 |
| 30-05-2019 | GabrielA | ID009 | 224 | None | 10:1 q weight, GAP | 2.4695 / 2.1865 |

## Validation Losses by Architecture

### DenseNet121
| Variant | Resolution | Total Loss | Geodesic Loss | R2 Loss | Geodesic Angular | MSE Loss | Explained Variance |
|---------|-----------|-----------|--------------|---------|-----------------|---------|-------------------|
| WO Reg | 300 | 0.2636 | 0.0441 | 0.1613 | 2.2574 | 0.0042 | 1.7010 |
| WO Reg - Reduce LR - No dropout - LSTM | 300 | 0.4009 | 0.0931 | 0.1592 | 3.1573 | 0.0040 | 2.9262 |

### Inceptionv3
| Variant | Resolution | Total Loss | Geodesic Loss | R2 Loss | Geodesic Angular | MSE Loss | Explained Variance |
|---------|-----------|-----------|--------------|---------|-----------------|---------|-------------------|
| Standard | 300 | 0.1154 | 0.0193 | 0.0641 | 4.1073 | 0.0016 | - |
| No pooling | 300 | 0.4198 | 0.0559 | 0.2888 | 4.2577 | 0.0075 | - |
| Global avg pooling | 300 | 0.3189 | 0.0529 | 0.1986 | 4.0965 | 0.0050 | - |
| Production 600 (2nd r-loss) | 600 | -3.2585 (val) | QLoss: 0.0432 | RLoss: 0.1388 | RTwoLoss: 0.0754 | - | sx:-1.069 sq:-2.292 |

### ResNet50
| Variant | Resolution | Total Loss | Geodesic Loss | R2 Loss | Geodesic Angular | MSE Loss | Explained Variance |
|---------|-----------|-----------|--------------|---------|-----------------|---------|-------------------|
| WO Reg | 300 | 0.3197 | 0.0755 | 0.1586 | 3.4426 | 0.0040 | - |

### Rotation-only augmentation (600res)
| Experiment | Total Loss | Geodesic Loss | R2 Loss | Geodesic Angular | MSE Loss | Explained Variance |
|-----------|-----------|--------------|---------|-----------------|---------|-------------------|
| B) 600res rotation only | 1.0742 | 0.1174 | 0.0022 | 3.0691 | 0.0851 | 0.6591 |
| 1) 300res rotation only | 0.6221 | 0.5596 | 0.0588 | 3.1697 | 2.3627 | 0.7974 |

### Rotation & Translation Augmentation (300res)
| Experiment | Total Loss | Geodesic Loss | MSE Loss | Geodesic Angular | R2 Loss | Explained Variance | LB Score |
|-----------|-----------|--------------|---------|-----------------|---------|-------------------|---------|
| 2) R & t Aug | 0.2604 | 0.0719 | 0.1861 | 3.2294 | 0.0046 | 0.8058 | 0.4536 |
| 2) I_R & t Aug | 90.9278 | 0.0676 | 0.1410 | 3.1979 | 0.0035 | - | - |
| 3) R & t Aug, 2x q reweight | 94.4118 | 0.0564 | 0.1514 | 3.0615 | 0.0038 | 1.1061 | 0.4070 |
| 4) Wrong tensor shape, no reg | 0.3115 | 0.0522 | 0.1944 | 4.1052 | 0.0050 | 1.6956 | 0.3977 |
| 5) With regularization | 0.4656 | 0.0841 | 0.2832 | 3.0393 | 0.0071 | 0.5649 | - |

## 2020 Investigations
| Model | Epochs | Resolution | Best Val Loss |
|-------|--------|-----------|--------------|
| CBAM Inceptionv3 | 200 | 600 | saved |
| EfficientNetB3 | 200 | 300 | saved |
| EfficientNetB4 | 200 | 600 | - |
| Inceptionv3 (1200 dropout) | 200? | 600 | saved |
| Inceptionv3 (base) | 200 | 600 | saved |
| Inceptionv3 + CBAM | 200 | 600 | - |
| Incepv3_1200 attempt2 | - | 600 | saved |
| Incepv3_1200 float16 test | - | 600 | - |
| Incepv3_1200 ResolutionHead | - | 600 | - |
| Incepv3 CBAMatEnd | - | 600 | saved |
| Iv3_1200 bnatend | - | 600 | saved |
