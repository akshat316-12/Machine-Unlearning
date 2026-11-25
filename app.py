import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from engine import (
    run_first_order_unlearning, 
    run_second_order_lissa_fc, 
    evaluate_metrics
)

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend dashboard can communicate with Flask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Dataset & Model Initialization (Scaled to 5,000 samples) ---
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Scaled 5,000 sample partitions: 4,500 Retain, 500 Forget, 1,000 Test
retain_loader = DataLoader(Subset(trainset, list(range(0, 4500))), batch_size=64, shuffle=True)
forget_loader = DataLoader(Subset(trainset, list(range(4500, 5000))), batch_size=64, shuffle=True)
test_loader = DataLoader(Subset(testset, list(range(0, 1000))), batch_size=64, shuffle=False)

def create_model():
    m = torchvision.models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 10)
    for module in m.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    return m.to(device)

BASELINE_PATH = "baseline_5k.pth"
model = create_model()

if os.path.exists(BASELINE_PATH):
    print(f"Loading cached baseline model weights from {BASELINE_PATH}...")
    model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
    print("Baseline 5k model loaded successfully!")
else:
    print("Training baseline ResNet-18 model on 5,000 images (25 epochs on GPU)...")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    combined_loader = DataLoader(Subset(trainset, list(range(0, 5000))), batch_size=64, shuffle=True)
    for epoch in range(25):
        for inputs, targets in combined_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), BASELINE_PATH)
    print(f"Baseline training complete! Saved to {BASELINE_PATH}. Server starting...")

@app.route('/status', methods=['GET'])
def status():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return jsonify({
        "status": "active", 
        "device": str(device), 
        "device_name": device_name,
        "model": "ResNet-18", 
        "dataset": "CIFAR-10",
        "total_samples": 5000,
        "retain_samples": 4500,
        "forget_samples": 500
    })

@app.route('/unlearn/first-order', methods=['POST'])
def trigger_first_order():
    data = request.get_json(silent=True) or {}
    epochs = int(data.get('epochs', 3))
    lr = float(data.get('lr', 0.015))
    
    global model
    model = run_first_order_unlearning(model, retain_loader, forget_loader, test_loader, device, epochs=epochs, lr=lr)
    metrics = evaluate_metrics(model, retain_loader, forget_loader, test_loader, device)
    
    return jsonify({"message": f"First-order unlearning executed (epochs={epochs}, lr={lr})", "metrics": metrics})

@app.route('/unlearn/second-order', methods=['POST'])
def trigger_second_order():
    data = request.get_json(silent=True) or {}
    damping = float(data.get('damping', 0.05))
    
    global model
    model = run_second_order_lissa_fc(model, retain_loader, forget_loader, test_loader, device, damping=damping)
    metrics = evaluate_metrics(model, retain_loader, forget_loader, test_loader, device)
    
    return jsonify({"message": f"Second-order LiSSA unlearning executed on model.fc (damping={damping})", "metrics": metrics})

@app.route('/reset', methods=['POST'])
def reset_model():
    global model
    if os.path.exists(BASELINE_PATH):
        model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
        metrics = evaluate_metrics(model, retain_loader, forget_loader, test_loader, device)
        return jsonify({"message": "Model successfully reset to clean 5,000-sample baseline.", "metrics": metrics})
    else:
        return jsonify({"error": f"No {BASELINE_PATH} file found to reset from."}), 400

@app.route('/evaluate', methods=['GET'])
def get_evaluation():
    metrics = evaluate_metrics(model, retain_loader, forget_loader, test_loader, device)
    return jsonify({"current_metrics": metrics})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Machine Unlearning Capstone Engine is running!",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "available_endpoints": {
            "status": "/status",
            "first_order_unlearning": "/unlearn/first-order (POST)",
            "second_order_unlearning": "/unlearn/second-order (POST)",
            "reset_model": "/reset (POST)",
            "evaluate_metrics": "/evaluate (GET)"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
