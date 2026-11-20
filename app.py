from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from engine import run_first_order_unlearning, run_second_order_lissa_fc, evaluate_metrics

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend dashboard can communicate with Flask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset & Model Initialization ---
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Streamlined subsets for fast prototype testing
retain_loader = DataLoader(Subset(trainset, list(range(0, 400))), batch_size=32, shuffle=True)
forget_loader = DataLoader(Subset(trainset, list(range(400, 500))), batch_size=32, shuffle=True)
test_loader = DataLoader(Subset(testset, list(range(0, 200))), batch_size=32, shuffle=False)

# Initialize ResNet-18 model for CIFAR-10
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Disable in-place activations to fix multi-forward backward pass error
for m in model.modules():
    if isinstance(m, torch.nn.ReLU):
        m.inplace = False

model = model.to(device)

# --- BASELINE TRAINING STEP (Boost epochs so the model properly memorizes) ---
print("Training baseline ResNet-18 model on subset so there is data to unlearn...")
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

combined_loader = DataLoader(Subset(trainset, list(range(0, 500))), batch_size=32, shuffle=True)
for epoch in range(10):  # Increased from 2 to 10 epochs for solid memorization
    for inputs, targets in combined_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
print("Baseline training complete! Model has memorized the data. Server starting...")

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "active", "device": str(device), "model": "ResNet-18", "dataset": "CIFAR-10"})

@app.route('/unlearn/first-order', methods=['POST'])
def trigger_first_order():
    data = request.json or {}
    epochs = int(data.get('epochs', 1))
    lr = float(data.get('lr', 1e-4))
    
    global model
    model = run_first_order_unlearning(model, retain_loader, forget_loader, device, epochs=epochs, lr=lr)
    metrics = evaluate_metrics(model, retain_loader, forget_loader, test_loader, device)
    
    return jsonify({"message": "First-order unlearning executed", "metrics": metrics})

@app.route('/unlearn/second-order', methods=['POST'])
def trigger_second_order():
    data = request.json or {}
    damping = float(data.get('damping', 0.01))
    
    global model
    model = run_second_order_lissa_fc(model, retain_loader, forget_loader, device, damping=damping)
    metrics = evaluate_metrics(model, retain_loader, forget_loader, test_loader, device)
    
    return jsonify({"message": "Second-order LiSSA unlearning executed on model.fc", "metrics": metrics})

@app.route('/evaluate', methods=['GET'])
def get_evaluation():
    metrics = evaluate_metrics(model, retain_loader, forget_loader, test_loader, device)
    return jsonify({"current_metrics": metrics})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Machine Unlearning Capstone API is running!",
        "available_endpoints": {
            "status": "/status",
            "first_order_unlearning": "/unlearn/first-order (POST)",
            "second_order_unlearning": "/unlearn/second-order (POST)",
            "evaluate_metrics": "/evaluate (GET)"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)