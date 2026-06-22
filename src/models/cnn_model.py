import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes=2, input_length=200):
        super().__init__()
        
        # Calculate the size after conv and pooling operations
        # After Conv1d and MaxPool1d(2) twice: input_length / 4
        self.input_length = input_length
        flattened_size = 32 * (input_length // 4)

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)


class CNNEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv = model.conv
        self.flatten = model.flatten
        self.embedding = model.fc[0]

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.embedding(x)