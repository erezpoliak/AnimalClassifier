import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    """
    A customizable CNN class.
    Parameters:
    - conv_layers: List specifying number of output channels for each convolutional layer.
    - fc_layers: List specifying number of neurons for each fully connected layer.
    - input_size: Size of the input images (assumed square).
    - n_classes: Number of output classes.
    - dropout: Dropout rate (0 means no dropout).
    - batch_norm: Boolean indicating whether to use batch normalization.
    - pool_after: After how many layers to pool
    """
    def __init__(self, conv_layers, fc_layers, input_size = 224, n_classes = 15, dropout = 0, batch_norm = False, pool_after=1):
        super(CustomCNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_after = pool_after

        # Build convolutional layers
        in_channels = 3
        kernel_size = 3
        padding = 1
        stride = 1
        for out_channels in conv_layers:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm))
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        
        # Calculate output size after conv + pool layers
        h, w = input_size, input_size
        curr_pool_after_count = self.pool_after
        for _ in conv_layers:
            curr_pool_after_count -= 1
            if curr_pool_after_count == 0:
              h = h // 2
              w = w // 2
              curr_pool_after_count = self.pool_after
        flattened_size = conv_layers[-1] * h * w

        # Build fully connected layers
        in_size = flattened_size
        for out_size in fc_layers:
            self.fc_layers.append(nn.Linear(in_size, out_size))
            in_size = out_size
        self.output_layer = nn.Linear(in_size, n_classes) 

        # He init for conv and fc layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        pool_counter = self.pool_after
        for i, conv in enumerate(self.conv_layers):
            pool_counter -= 1
            x = conv(x)
            if self.bn_layers:
                x = self.bn_layers[i](x)
            x = F.relu(x)
            if pool_counter == 0:
                x = self.pool(x)
                pool_counter = self.pool_after
        
        x = x.view(x.size(0), -1)  # Flatten
        for fc in self.fc_layers:
            x = fc(x)
            x = F.relu(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.output_layer(x)
        return x
                        