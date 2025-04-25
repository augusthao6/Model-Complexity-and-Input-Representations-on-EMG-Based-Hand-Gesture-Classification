class middleLSTM(nn.Module):
    def __init__(self, input_channel, num_classes=10):
        super(middleLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size = input_channel,
            hidden_size=64,
            num_layers =1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout1=nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size = 128,
            hidden_size=64,
            num_layers =1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout2=nn.Dropout(0.3)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, 128)

        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        lstm_out1, _ =self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)

        lstm_out2, _ =self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)


        x = self.global_avg_pool(lstm_out2.transpose(1, 2)).squeeze(-1)

        x = F.relu(self.fc(x))
        x = self.dropout2(x)

        out = self.out(x)

        return out