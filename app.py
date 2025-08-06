from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

app = Flask(__name__)
CORS(app)

class Net(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super().__init__()
        layers = []
        last_size = input_size
        for neurons in hidden_layers:
            layers.append(nn.Linear(last_size, neurons))
            layers.append(nn.ReLU())
            last_size = neurons
        layers.append(nn.Linear(last_size, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

@app.route('/train', methods=['POST'])
def train():
    params = request.json
    layers = params.get("layers", [4,3,2])
    epochs = int(params.get("epochs", 200))
    lr = float(params.get("lr", 0.01))
    data = params.get("data")
    if data and "X" in data and "y" in data:
        X = np.array(data["X"], dtype=np.float32)
        y = np.array(data["y"], dtype=np.int64)
    else:
        # Synthetic XOR data
        X = np.random.randn(200, 2)
        y = ((X[:,0]>0) ^ (X[:,1]>0)).astype(int)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    net = Net(input_size=X.shape[1], hidden_layers=layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    losses = []
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
        "final_loss": losses[-1],
        "accuracy": accuracy,
        "losses": losses,
        "predictions": preds.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)