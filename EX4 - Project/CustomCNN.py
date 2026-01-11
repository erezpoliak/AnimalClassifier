# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # Defining flexible CNN for experiments
# class CustomCNN(nn.Module):
#   def __init__(self, conv_channels, fc_hidden, input_size = 224, use_batch_norm = False, dropout = 0, dropout_conv = 0, num_classes = 15):
#     super(CustomCNN, self).__init__()
    
#     self.conv_layers = nn.ModuleList()
#     self.bn_layers = nn.ModuleList() if use_batch_norm else None
#     self.dropout_conv_layers = nn.ModuleList() if dropout_conv > 0 else None
#     self.pool = nn.MaxPool2d(2,2)

#     # Conv layers
#     in_channels = 3
#     for out_channels in conv_channels:
#       self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
      
#       # Batch normalization
#       if use_batch_norm:
#         self.bn_layers.append(nn.BatchNorm2d(out_channels))
      
#       # Spatial Dropout
#       if dropout_conv > 0:
#         self.dropout_conv_layers.append(nn.Dropout2d(dropout_conv))
      
#       in_channels = out_channels
    
#     h, w = input_size, input_size
#     for _ in conv_channels:
#       h = h // 2
#       w = w // 2
#     flattend_size = conv_channels[-1] * h * w

#     # Fully connected layers
#     self.fc1 = nn.Linear(flattend_size, fc_hidden)
#     self.dropout = nn.Dropout(dropout) if dropout > 0 else None
#     self.fc2 = nn.Linear(fc_hidden, num_classes)

#     # kaiming He init
#     self._init_weights()
  
#   def forward(self,x):
#     # Conv layers
#     for i, conv_layer in enumerate(self.conv_layers):
#       x = conv_layer(x)
      
#       # Batch normalization
#       if self.bn_layers is not None:
#         x = self.bn_layers[i](x)
      
#       x = F.relu(x)
#       x = self.pool(x)
      
#       # Spacial dropout
#       if self.dropout_conv_layers is not None:
#         x = self.dropout_conv_layers[i](x)
    
#     # Fully connected layers
#     x = torch.flatten(x, 1)
#     x = F.relu(self.fc1(x))
#     if self.dropout:
#       x = self.dropout(x)
#     x = self.fc2(x)
#     return x
  
#   def _init_weights(self):
#     for m in self.modules():
#       if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#           nn.init.zeros_(m.bias)
#       elif isinstance(m, nn.Linear):
#         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#           nn.init.zeros_(m.bias)
#       elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#         nn.init.ones_(m.weight)
#         nn.init.zeros_(m.bias)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    """
    Versatile CNN class for experimentation with different architectures,
    normalization, and regularization techniques.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
        architecture (str): Architecture type - 'shallow', 'medium', 'deep', 'custom'
        use_batch_norm (bool): Whether to use batch normalization
        dropout_rate (float): Dropout probability (0 for no dropout)
        custom_config (dict): Custom architecture configuration (optional)
        
    Custom config format (example):
    {
        'conv_layers': [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ],
        'fc_layers': [512, 256]  # Hidden layer sizes
    }
    """
    
    def __init__(self, num_classes, input_channels=3, architecture='medium',
                 use_batch_norm=False, dropout_rate=0.0, custom_config=None):
        super(CustomCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.architecture = architecture
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Get architecture configuration
        if architecture == 'custom' and custom_config is not None:
            config = custom_config
        else:
            config = self._get_predefined_config(architecture)
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_ch = input_channels
        for i, conv_config in enumerate(config['conv_layers']):
            # Convolutional layer
            conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=conv_config['out_channels'],
                kernel_size=conv_config['kernel_size'],
                stride=conv_config.get('stride', 1),
                padding=conv_config.get('padding', 1)
            )
            self.conv_layers.append(conv)
            
            # Batch normalization (optional)
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(conv_config['out_channels']))
            else:
                self.bn_layers.append(nn.Identity())
            
            # Pooling layer
            pool_type = conv_config.get('pool', 'max')
            pool_size = conv_config.get('pool_size', 2)
            if pool_type == 'max':
                self.pool_layers.append(nn.MaxPool2d(pool_size))
            elif pool_type == 'avg':
                self.pool_layers.append(nn.AvgPool2d(pool_size))
            else:
                self.pool_layers.append(nn.Identity())
            
            in_ch = conv_config['out_channels']
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened size
        self.flatten_size = config['conv_layers'][-1]['out_channels'] * 4 * 4
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        fc_sizes = [self.flatten_size] + config['fc_layers'] + [num_classes]
        for i in range(len(fc_sizes) - 1):
            self.fc_layers.append(nn.Linear(fc_sizes[i], fc_sizes[i + 1]))
            if i < len(fc_sizes) - 2:  # Don't add dropout after last layer
                self.dropout_layers.append(nn.Dropout(dropout_rate))
            else:
                self.dropout_layers.append(nn.Identity())
    
    def _get_predefined_config(self, architecture):
        """Get predefined architecture configurations"""
        configs = {
            'shallow': {
                'conv_layers': [
                    {'out_channels': 32, 'kernel_size': 5, 'padding': 2, 'pool': 'max', 'pool_size': 2},
                    {'out_channels': 64, 'kernel_size': 5, 'padding': 2, 'pool': 'max', 'pool_size': 2}
                ],
                'fc_layers': [128]
            },
            'medium': {
                'conv_layers': [
                    {'out_channels': 32, 'kernel_size': 3, 'padding': 1, 'pool': 'max', 'pool_size': 2},
                    {'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'pool': 'max', 'pool_size': 2},
                    {'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'pool': 'max', 'pool_size': 2}
                ],
                'fc_layers': [256, 128]
            },
            'deep': {
                'conv_layers': [
                    {'out_channels': 32, 'kernel_size': 3, 'padding': 1, 'pool': 'max', 'pool_size': 2},
                    {'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'pool': 'max', 'pool_size': 2},
                    {'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'pool': 'max', 'pool_size': 2},
                    {'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'pool': 'max', 'pool_size': 2},
                    {'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'pool': 'max', 'pool_size': 2}
                ],
                'fc_layers': [512, 256, 128]
            }
        }
        return configs[architecture]
    
    def forward(self, x):
        # Convolutional layers with activation and pooling
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = pool(x)
        
        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        for i, (fc, dropout) in enumerate(zip(self.fc_layers, self.dropout_layers)):
            x = fc(x)
            if i < len(self.fc_layers) - 1:  # Don't apply ReLU after last layer
                x = F.relu(x)
            x = dropout(x)
        
        return x
    
    def get_num_parameters(self):
        """Return the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
    
    def get_architecture_summary(self):
        """Return a string summary of the architecture"""
        summary = f"CustomCNN Architecture:\n"
        summary += f"  Input channels: {self.input_channels}\n"
        summary += f"  Architecture type: {self.architecture}\n"
        summary += f"  Batch Normalization: {self.use_batch_norm}\n"
        summary += f"  Dropout rate: {self.dropout_rate}\n"
        summary += f"  Total parameters: {self.get_num_parameters():,}\n"
        summary += f"\nConvolutional Layers:\n"
        for i, conv in enumerate(self.conv_layers):
            summary += f"  Conv{i+1}: {conv.in_channels} -> {conv.out_channels}, "
            summary += f"kernel={conv.kernel_size[0]}x{conv.kernel_size[1]}\n"
        summary += f"\nFully Connected Layers:\n"
        for i, fc in enumerate(self.fc_layers):
            summary += f"  FC{i+1}: {fc.in_features} -> {fc.out_features}\n"
        return summary
