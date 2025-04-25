import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        #Calculate average and max pool for each channel - "Squeeze"
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        #Put the pools through an MLP - "Excitation"
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1) #Add average and max out and apply sigmoid - gives us channel attention weights
        return x * channel_att #Apply channel attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2) #Perform 1d convolution - gives us time domain
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1) #Concatenate avg and max out

        attention_map = self.sigmoid(self.conv(attention_input)) #Perform convolution and sigmoid, giving us temporal attention weights

        return x * attention_map #Apply temporal attention

class middleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, dropout_rate=0.3):
        super(middleCNN, self).__init__()

        #First convolution block
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.channel_attention = ChannelAttention(32) #Add Channel Attention after first convolution block

        #Second convolution block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.spatial_attention = SpatialAttention() #Add Spatial (Temporal) Attention after second convolution block
        self.dropout1 = nn.Dropout(dropout_rate)

        #Third convolution block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)

        #Global pooling and FC layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        #First convolution block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.channel_attention(x)

        #Second convolution block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.spatial_attention(x)
        x = self.dropout1(x)

        #Third convolution block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout2(x)

        #Global pooling and FC layers
        x = self.global_avg_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.out(x)
        return x
