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
  print("Corrupted images found:", len(corrupted))

def load_dataset(data_path):
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
  dataset = datasets.ImageFolder(root=data_path, transform=transform)
  return dataset

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