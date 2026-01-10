import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, optimizer, epochs, patience, device, weight_decay = 0):
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()

  # apply weight decay if provided
  if weight_decay > 0:
    for g in optimizer.param_groups:
      g['weight_decay'] = weight_decay

  history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
  best_loss = float('inf')
  patience_counter = 0
  best_model_state = None

  for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * inputs.size(0)
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    # Validation
    val_loss, val_acc = evaluate(model, val_loader, device)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if (epoch + 1) % 5:
      print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')

    # Early Stopping
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_state = model.state_dict()
      patience_counter = 0
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        model.load_state_dict(best_model_state)
        return model, history

  model.load_state_dict(best_model_state)
  return model, history


def evaluate(model, val_loader, device):
  model.eval()
  model.to(device)
  criterion = nn.CrossEntropyLoss()

  running_loss = 0
  correct = 0
  total = 0

  with torch.no_grad():
    for inputs, labels in val_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      running_loss = loss.item() * inputs.size(0)
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()

  avg_loss = running_loss / total
  accuracy = correct / total
  return avg_loss, accuracy

def plot_training_curves(history):
  epochs = range(1, len(history['train_loss']) + 1)
  title = 'Training Curves'

  # Plot Loss
  plt.figure(figsize=(12,5))
  plt.subplot(1,2,1)
  plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
  plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
  plt.title(f"{title} - Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.grid(True)

  # Plot Accuracy
  plt.subplot(1,2,2)
  plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
  plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
  plt.title(f"{title} - Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()