class middleCNNLSTM(nn.Module):
    def __init__(self, input_channels, seq_length, num_classes, hidden_dim=64, dropout_rate=0.3):
        super(middleCNNLSTM, self).__init__()

        #CNN Feature Extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim*2,  #2x for bidirectional
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout2 = nn.Dropout(dropout_rate)

        #Final classification
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        #CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.permute(0, 2, 1)

        #First LSTM layer
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)

        #Second LSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)

        #Use the last time step for classification
        x = lstm_out2[:, -1, :]

        #Final classification
        x = self.fc(x)

        return x