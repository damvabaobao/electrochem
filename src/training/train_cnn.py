import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.cnn_model import CNNModel
from src.data.dataset import SignalDataset
from sklearn.preprocessing import LabelEncoder

def train_cnn():

    data = pd.read_csv("data/processed/dataset.csv")
    le = LabelEncoder()
    data["label"] = le.fit_transform(data["label"])
    

    X = data.drop("label", axis=1).values
    y = data["label"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_loader = DataLoader(SignalDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(SignalDataset(X_val, y_val), batch_size=32)

    num_classes = len(set(y))
    print("Num classes: ", num_classes)
    model = CNNModel(num_classes = num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")

    for epoch in range(30):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += criterion(model(x), y).item()

        print(epoch, train_loss, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/cnn/cnn_best.pth")

    print("CNN DONE")
if __name__ == "__main__":
    train_cnn()