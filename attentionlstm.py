class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        #get Q, K, V projections, scale and normalize
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        #Compute query, key, value projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        #Compute attention scores
        energy = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        #Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        #Apply attention weights to values
        context = torch.matmul(attention, v)

        #Apply residual connection and normalization
        output = self.norm(x + context)

        return output, attention

class ChannelAttention(nn.Module):
    def __init__(self, channel_dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_dim // reduction, channel_dim, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1))
        channel_att = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * channel_att

class middleLSTM(nn.Module):
    def __init__(self, input_channel, num_classes=10):
        super(middleLSTM, self).__init__()

        hidden_dim = 128
        bidirectional = True
        lstm_dim = hidden_dim // 2 if bidirectional else hidden_dim

        self.lstm1 = nn.LSTM(
            input_size=input_channel,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout1 = nn.Dropout(0.3)

        #Channel attention after first LSTM layer
        self.channel_attention = ChannelAttention(hidden_dim)

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout2 = nn.Dropout(0.3)

        #Temporal attention after second LSTM layer
        self.temporal_attention = TemporalAttention(hidden_dim)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)

        #Apply channel attention - first transpose to get channels in middle
        channel_att_in = lstm_out1.transpose(1, 2) 
        channel_att_out = self.channel_attention(channel_att_in)
        lstm_out1 = channel_att_out.transpose(1, 2) 

        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)

        #Apply temporal attention
        temporal_out, _ = self.temporal_attention(lstm_out2)

        #Global pooling
        x = temporal_out.transpose(1, 2)
        x = self.global_avg_pool(x).squeeze(-1)

        x = F.relu(self.fc(x))
        x = self.dropout2(x)
        out = self.out(x)

        return out