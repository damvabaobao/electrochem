import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

from src.models.cnn_model import CNNModel, CNNEncoder
from src.models.attention_model import AttentionModel
from src.features.feature_extraction import extract_features
from src.data.preprocess import normalize, pad_signal

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score)


def build_dataset(signals, labels): # encoder

    X_feat = []
    X_deep = []
    y = []

    for signal, label in zip(signals, labels):
        signal = normalize(signal)
        signal = pad_signal(signal, 200)

        # handcrafted
        feat = extract_features(signal)

        # deep
        s = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            emb = encoder(s).numpy().flatten()

        X_feat.append(feat)
        X_deep.append(emb)
        y.append(label)

    return np.array(X_feat), np.array(X_deep), np.array(y)

criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []

train_accs = []
val_accs = []


def train_attention():

    data = pd.read_csv(r"D:\electrochem_ai(1)\physics_dataset.csv")
    le = LabelEncoder()
    data["label"] = le.fit_transform(data["label"])

    signals = data.drop("label", axis=1).values
    labels = data["label"].values

    # load CNN encoder
    #num_classes = len(set(labels))
    #cnn = CNNModel(num_classes=num_classes)
    #cnn.load_state_dict(torch.load(r"D:\electrochem_ai(1)\models_save\cnn\cnn_best.pth"))
    #cnn.eval()

    #encoder = CNNEncoder(cnn)
    #encoder.eval()

    # build dataset
    X_feat, X_deep, y = build_dataset(signals, labels) # encoder

    # split
    #Xf_train, Xf_test, Xd_train, Xd_test, y_train, y_test = train_test_split(X_feat, X_deep, y, test_size=0.2, random_state=42)

    Xf_train, Xf_temp, Xd_train, Xd_temp, y_train, y_temp = train_test_split(X_feat, X_deep, y, test_size=0.3, random_state=42, stratify=y)
    Xf_val, Xf_test, Xd_val, Xd_test, y_val, y_test = train_test_split(Xf_temp, Xd_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # tensor
    Xf_train = torch.tensor(Xf_train, dtype=torch.float32)
    Xd_train = torch.tensor(Xd_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    Xf_val = torch.tensor(Xf_val, dtype=torch.float32)
    Xd_val = torch.tensor(Xd_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)


    Xf_test = torch.tensor(Xf_test, dtype=torch.float32)
    Xd_test = torch.tensor(Xd_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(Xf_train, Xd_train, y_train),
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(TensorDataset(Xf_val, Xd_val, y_val),
                            batch_size = 32,
                            shuffle=False)

    # model
    model = AttentionModel(
        feat_dim=X_feat.shape[1],
        deep_dim=X_deep.shape[1],
        num_classes=num_classes
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # training
    for epoch in range(30):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for f,d,yb in train_loader:
            optimizer.zero_grad()
            out = model(f,d)
            loss = criterion(out,yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(out,1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for f, d, yb in val_loader:
                out = model(f, d)
                loss = criterion(out, yb)
                val_loss += loss.item()
                _, pred = torch.max(out, 1)
                total += yb.size(0)
                correct += (pred == yb).sum().item()
    
        val_loss /= len(val_loader)
        val_acc = correct / total

    # save
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
        f"Epoch {epoch+1}/30 "
        f"Train Loss={train_loss:.4f} "
        f"Val Loss={val_loss:.4f} "
        f"Train Acc={train_acc:.4f} "
        f"Val Acc={val_acc:.4f}"
        )


    torch.save(model.state_dict(), "models/attention/attention.pth")

    print("ATTENTION MODEL DONE")
    test_loader = DataLoader(
    TensorDataset(Xf_test,Xd_test,y_test),
    batch_size=32,
    shuffle=False
)
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():

        for f,d,yb in test_loader:

            out = model(f,d)

            _, pred = torch.max(out,1)

            y_true.extend(yb.numpy())

            y_pred.extend(pred.numpy())

    acc = accuracy_score(y_true,y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("\n========== ATTENTION ==========")
    print("Accuracy :",acc)
    print("Precision:",precision)
    print("Recall   :",recall)
    print("F1 Score :",f1)

    plt.figure(figsize=(8,5))

    plt.plot(train_losses,label="Train")

    plt.plot(val_losses,label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Attention Model Loss")

    plt.legend()

    plt.savefig(
        "results/attention_loss_curve.png",
        dpi=300
    )

    plt.show()

    plt.figure(figsize=(8,5))

    plt.plot(train_accs,label="Train")

    plt.plot(val_accs,label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Attention Model Accuracy")

    plt.legend()

    plt.savefig(
        "results/attention_accuracy_curve.png",
        dpi=300
    )

    plt.show()

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=le.classes_
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Attention Confusion Matrix")

    plt.savefig(
        "results/attention_confusion_matrix.png",
        dpi=300
    )
    plt.show()
if __name__ == "__main__":
    train_attention()