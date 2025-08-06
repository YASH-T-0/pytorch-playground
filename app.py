from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

app = Flask(__name__)
CORS(app)

# --- Model Definitions ---

class ANN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=2):
        super().__init__()
        layers = []
        last_size = input_size
        for neurons in hidden_layers:
            layers.append(nn.Linear(last_size, neurons))
            layers.append(nn.ReLU())
            last_size = neurons
        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=1, num_classes=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq, feature]
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Training Route ---
@app.route('/train', methods=['POST'])
def train():
    params = request.json
    model_type = params.get("model_type", "ANN").upper()  # 'ANN', 'CNN', or 'RNN'
    layers = params.get("layers", [4,3,2])
    epochs = int(params.get("epochs", 200))
    lr = float(params.get("lr", 0.01))
    data = params.get("data")
    losses = []
    preds = []
    accuracy = 0.0

    # --- Data Preprocessing & Model Selection ---
    if model_type == "CNN":
        # Expecting images: data["X"] shape (N, 1, 28, 28), y: (N,)
        if data and "X" in data and "y" in data:
            X = np.array(data["X"], dtype=np.float32)
            y = np.array(data["y"], dtype=np.int64)
        else:
            # Dummy MNIST-like data
            X = np.random.randn(100, 1, 28, 28)
            y = np.random.randint(0, 2, 100)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        net = CNN(input_channels=1, num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = net(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        preds = torch.argmax(net(X_tensor), dim=1).numpy()
        accuracy = float(np.mean(preds == y))

    elif model_type == "RNN":
        # Expecting sequences: data["X"] shape (N, seq_len, feature), y: (N,)
        if data and "X" in data and "y" in data:
            X = np.array(data["X"], dtype=np.float32)
            y = np.array(data["y"], dtype=np.int64)
        else:
            # Dummy sequence data: batch=100, seq_len=10, features=4
            X = np.random.randn(100, 10, 4)
            y = np.random.randint(0, 2, 100)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        net = RNN(input_size=X.shape[2], hidden_size=16, num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = net(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        preds = torch.argmax(net(X_tensor), dim=1).numpy()
        accuracy = float(np.mean(preds == y))

    else:  # Default to ANN
        if data and "X" in data and "y" in data:
            X = np.array(data["X"], dtype=np.float32)
            y = np.array(data["y"], dtype=np.int64)
        else:
            # Synthetic XOR data
            X = np.random.randn(200, 2)
            y = ((X[:,0]>0) ^ (X[:,1]>0)).astype(int)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        net = ANN(input_size=X.shape[1], hidden_layers=layers, output_size=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = net(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        preds = torch.argmax(net(X_tensor), dim=1).numpy()
        accuracy = float(np.mean(preds == y))

    return jsonify({
        "final_loss": float(losses[-1]) if losses else None,
        "accuracy": accuracy,
        "losses": [float(l) for l in losses],
        "predictions": preds.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)