# model.py
import torch
import torch.nn as nn

# Define BiLSTM Poetry Model with BiLSTM Layers
class BiLSTMPoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(BiLSTMPoetryModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # First BiLSTM layer
        self.bilstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # Output layer

    def forward(self, x):
        x = self.embedding(x)
        # First BiLSTM layer
        lstm_out, _ = self.bilstm1(x)
        # Fully connected layer
        output = self.fc(lstm_out[:, -1, :])
        return output
