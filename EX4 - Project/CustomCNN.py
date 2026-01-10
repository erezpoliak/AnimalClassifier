import torch
import torch.nn as nn
import torch.nn.functional as F


# Defining flexible CNN for experiments
class CustomCNN(nn.Module):
  def __init__(self, conv_channels, fc_hidden, input_size = 224, use_batch_norm = False, dropout = 0, dropout_conv = 0, num_classes = 15):
    super(CustomCNN, self).__init__()
    
    self.conv_layers = nn.ModuleList()
    self.bn_layers = nn.ModuleList() if use_batch_norm else None
    self.dropout_conv_layers = nn.ModuleList() if dropout_conv > 0 else None
    self.pool = nn.MaxPool2d(2,2)

    # Conv layers
    in_channels = 3
    for out_channels in conv_channels:
      self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
      
      # Batch normalization
      if use_batch_norm:
        self.bn_layers.append(nn.BatchNorm2d(out_channels))
      
      # Spatial Dropout
      if dropout_conv > 0:
        self.dropout_conv_layers.append(nn.Dropout2d(dropout_conv))
      
      in_channels = out_channels
    
    h, w = input_size, input_size
    for _ in conv_channels:
      h = h // 2
      w = w // 2
    flattend_size = conv_channels[-1] * h * w

    # Fully connected layers
    self.fc1 = nn.Linear(flattend_size, fc_hidden)
    self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    self.fc2 = nn.Linear(fc_hidden, num_classes)

    # kaiming He init
    self._init_weights()
  
  def forward(self,x):
    # Conv layers
    for i, conv_layer in enumerate(self.conv_layers):
      x = conv_layer(x)
      
      # Batch normalization
      if self.bn_layers is not None:
        x = self.bn_layers[i](x)
      
      x = F.relu(x)
      x = self.pool(x)
      
      # Spacial dropout
      if self.dropout_conv_layers is not None:
        x = self.dropout_conv_layers[i](x)
    
    # Fully connected layers
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    if self.dropout:
      x = self.dropout(x)
    x = self.fc2(x)
    return x
  
  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)