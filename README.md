# Animal Image Classification - Deep Learning

Multi-class animal classification with custom CNNs and transfer learning. Achieved **98.63% test accuracy** using ResNet50 fine-tuning.

## 📊 Dataset

- **15 animal classes**: Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra
- **~2000 images** (70/15/15 split)
- **224×224 resolution**

## 🎯 Results Summary

| Model                | Test Accuracy | Test Loss | Notes                             |
| -------------------- | ------------- | --------- | --------------------------------- |
| Custom CNN (scratch) | 93.86%        | 0.390     | 4 conv + regularization           |
| CIFAR-10 Transfer    | 91.81%        | 0.366     | Resolution mismatch limited gains |
| ResNet50 Frozen      | 98.29%        | 0.189     | Only FC trained                   |
| ResNet50 Partial     | 98.63%        | 0.039     | layer4 + FC trained               |
| **ResNet50 Full**    | **98.63%**    | **0.027** | **All layers trained**            |

## 🔬 Key Experiments

### Part 1: Custom CNN From Scratch

- **Architecture**: 4 conv [32,64,128,256] → 2 FC [512,256]
- **Optimizer**: Adam (LR 0.001)
- **Regularization**: Dropout (0.2) + Weight Decay (0.001) + Data Augmentation
- **Result**: 93.86% test accuracy

### Part 2: CIFAR-10 Transfer Learning

- **Pretraining**: CIFAR-10 (32×32)
- **Architecture**: Conv [64,64,128,128,256,256] -> FC [512,256], Pooling every other layer
- **Optimizer**: Adam (LR 0.001)
- **Regularization**: Dropout (0.1) + Weight Decay (0.001) + Data Augmentation
- **Fine-tuning**: LR 0.0001 on animal dataset
- **Result**: 91.81% test accuracy
- **Insight**: Resolution mismatch (32×32 → 224×224) limited transfer benefits

### Part 3: ResNet50 Transfer Learning

Three freezing strategies tested:

| Strategy            | Trainable Params | Test Acc   | Test Loss |
| ------------------- | ---------------- | ---------- | --------- |
| Frozen backbone     | 30k              | 98.29%     | 0.189     |
| Partial (layer4+FC) | 15M              | 98.63%     | 0.039     |
| Full fine-tuning    | 25M              | **98.63%** | **0.027** |

**Insight**: ImageNet alignment perfect → frozen backbone already 98%+

## 💡 Key Takeaways

1. **Transfer learning requires alignment**: ImageNet (224×224, animals) >> CIFAR-10 (32×32, generic)
2. **Data augmentation dominates**: +5.5% accuracy on small datasets
3. **Pre-trained depth wins**: ResNet50 (50 layers, ImageNet) beats custom 4-layer CNN
4. **Smart regularization > raw depth**: 4-layer CNN (95%) competitive with proper tuning

## 🛠️ Technologies

PyTorch • torchvision • NumPy • Matplotlib • Scikit-learn
