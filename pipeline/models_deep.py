"""Optional deep-learning models trained directly on raw epoch tensors
(channels x time), compared against the handcrafted-feature classical models.

EEGNet is the standard compact architecture for this exact input shape
(Lawhern et al. 2018); SimpleCNN and CNNLSTM are lighter baselines.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)

from pipeline import config


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # separable
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_dim = self.block2(self.block1(dummy)).numel()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)


class SimpleCNN(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (n_channels, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (1, 25), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_dim = self.net(dummy).numel()
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x.flatten(1))


class CNNLSTM(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (n_channels, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d((1, 4)),
        )
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        x = self.conv(x)              # (B, C=16, 1, T')
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T', 16)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


MODEL_CLASSES = {"EEGNet": EEGNet, "SimpleCNN": SimpleCNN, "CNNLSTM": CNNLSTM}


def train_and_eval(model_name, X_train, y_train, X_test, y_test, n_epochs=30,
                    lr=1e-3, batch_size=32, seed=config.SEED, device=None):
    """X_*: (n_epochs, n_channels, n_times) numpy arrays (volts or microvolts,
    should be roughly standardized). Returns (metrics_dict, model).
    """
    torch.manual_seed(seed)
    # Default to CPU: this is a shared GPU box and these models/datasets are
    # small enough that CPU training is fast and doesn't fight other jobs
    # for VRAM (we hit a CUDA OOM from other processes when defaulting to GPU).
    device = device or "cpu"

    mu, sigma = X_train.mean(), X_train.std() + 1e-12
    Xtr = torch.tensor((X_train - mu) / sigma, dtype=torch.float32).unsqueeze(1)
    Xte = torch.tensor((X_test - mu) / sigma, dtype=torch.float32).unsqueeze(1)
    ytr = torch.tensor(y_train, dtype=torch.long)
    yte = torch.tensor(y_test, dtype=torch.long)

    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    model = MODEL_CLASSES[model_name](n_channels, n_times).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                          generator=torch.Generator().manual_seed(seed))

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device)).cpu()
        proba = torch.softmax(logits, dim=1)[:, 1].numpy()
        pred = logits.argmax(dim=1).numpy()

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan,
    }
    return metrics, pred, proba, model
