class multiBranchCNN(nn.Module):
    def __init__(self, input_channels, input_freq, num_classes):
        super(multiBranchCNN, self).__init__()
        self.time_conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.time_bn1 = nn.BatchNorm1d(32)

        self.time_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.time_bn2 = nn.BatchNorm1d(64)

        self.time_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.time_bn3 = nn.BatchNorm1d(128)

        self.freq_conv1 = nn.Conv1d(input_freq, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.freq_bn1 = nn.BatchNorm1d(32)

        self.freq_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.freq_bn2 = nn.BatchNorm1d(64)

        self.freq_conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.freq_bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, num_classes)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

    def forward(self, x1, x2):
        x1 = F.relu(self.time_bn1(self.time_conv1(x1)))
        x1 = self.pool(x1)

        x1 = F.relu(self.time_bn2(self.time_conv2(x1)))
        x1 = self.pool(x1)
        x1 = self.dropout1(x1)

        x1 = F.relu(self.time_bn3(self.time_conv3(x1)))
        x1 = self.pool(x1)
        x1 = self.dropout2(x1)

        x1 = self.global_avg_pool(x1).squeeze(-1)

        x2 = F.relu(self.freq_bn1(self.freq_conv1(x2)))
        x2 = self.pool(x2)

        x2 = F.relu(self.freq_bn2(self.freq_conv2(x2)))
        x2 = self.pool(x2)
        x2 = self.dropout1(x2)

        x2 = F.relu(self.freq_bn3(self.freq_conv3(x2)))
        x2 = self.pool(x2)
        x2 = self.dropout2(x2)

        x2 = self.global_avg_pool(x2).squeeze(-1)

        x = torch.cat([x1,x2], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.out(x)

        return x