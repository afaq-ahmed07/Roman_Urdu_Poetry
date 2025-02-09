# model.py
import torch
import torch.nn as nn

# Define BiLSTM Poetry Model
class BiLSTMPoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(BiLSTMPoetryModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # Multiply hidden_dim by 2 due to bidirectionality

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.bilstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        return output
