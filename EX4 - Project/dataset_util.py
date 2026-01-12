import os
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

def check_corrupted_images(data_path):
  corrupted = []

  for root, dirs, files in os.walk(data_path):
    for file in files:
      if file.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(root, file)
        try:
          img = Image.open(path)
          img.verify()
        except:
          corrupted.append(path)
  return corrupted

def display_basic_info(dataset):
  print("Number of images:", len(dataset))
  print("Number of classes:", len(dataset.classes))
  print("Classes:", dataset.classes)

def display_class_distribution(dataset):
  labels = [label for _, label in dataset]
  counter = Counter(labels)
  for cls, count in counter.items():
    print(f"{dataset.classes[cls]}: {count}")

def display_image_samples(dataset):
  fig, axes = plt.subplots(3, 5, figsize=(15, 9))

  # pick 15 random samples
  random_samples = torch.randperm(len(dataset))[:15]

  for idx, i in enumerate(random_samples):
    img, label = dataset[i]
    axes[idx//5, idx%5].imshow(img.permute(1, 2, 0))
    axes[idx//5, idx%5].set_title(dataset.classes[label])
    axes[idx//5, idx%5].axis("off")

  plt.show()

def dataset_split(dataset, train_ratio, val_ratio):
  train_size = int(train_ratio * len(dataset))
  val_size = int(val_ratio * len(dataset))
  test_size = len(dataset) - train_size - val_size
  train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
  return train_dataset, val_dataset, test_dataset

def calculate_normalization_stats(dataset):
  """
  Calculate mean and std for each channel across the dataset.
  Returns: (mean, std) as tuples of 3 values (R, G, B)
  """
  loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
  
  data = next(iter(loader))[0]  # Get all images
  
  mean = data.mean(dim=[0, 2, 3])  # Mean across batch, height, width
  std = data.std(dim=[0, 2, 3])    # Std across batch, height, width
  
  return mean.tolist(), std.tolist()


def get_normalized_transforms(mean, std, input_size=224, augment=False):
  """
  Create transform pipeline with normalization.
  
  Parameters:
  - mean: tuple of 3 values for RGB channels
  - std: tuple of 3 values for RGB channels
  - input_size: target image size
  - augment: whether to apply data augmentation (for training)
  """
  if augment:
    return transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10), 
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
  else:
    return transforms.Compose([
      transforms.Resize((input_size, input_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std)
    ])