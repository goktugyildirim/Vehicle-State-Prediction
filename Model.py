import torch
import torch.nn as nn


class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, lstm_output_feature_size, num_lstm_layers, output_dim, use_gpu, device):
        super(Seq2SeqModel, self).__init__()

        self.device = device
        self.use_gpu = use_gpu

        # Hidden dimensions
        self.lstm_output_feature_size = lstm_output_feature_size

        # Number of hidden layers
        self.num_lstm_layers = num_lstm_layers

        # LSTM
        self.lstm = nn.LSTM(input_dim, lstm_output_feature_size, num_lstm_layers, batch_first=True)

        # Regularization
        self.bn1 = nn.BatchNorm1d(num_features=lstm_output_feature_size)
        self.dropout = nn.Dropout2d(0.25)

        # Readout layer
        self.fc = nn.Linear(lstm_output_feature_size, output_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_output_feature_size).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_output_feature_size).requires_grad_()

        if self.use_gpu:
            h0, c0 = h0.to(self.device), c0.to(self.device)

        # I need to detach as I am doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        #print("LSTM input shape:", x.shape)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        #print("LSTM output shape: ", out.shape)

        # Normalizing using only the last time stamp!
        out = self.bn1(out[:, -1, :])
        out = self.dropout(out)
        out = torch.nn.functional.relu(out)

        # Index hidden state of last time step
        # out.size() --> batch_size, seq_length, lstm_output_feature_size
        # out[:, -1, :] --> batch_size, lstm_output_feature_size --> just want last time step hidden states!
        #print("LSTM the last time step output size, also FC layer input shape:", out[:, -1, :].shape)
        out = self.fc(out)
        #print("FC layer output shape (batch size, output_dim=4*output_timestamp_length):", out.shape)
        # 4(t)*(dx, dy, vx, vy)
        # out.size() --> 12, 4*output_timestamp
        return out
