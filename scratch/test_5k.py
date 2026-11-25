import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# 5,000 Samples: 4,500 Retain, 500 Forget, 1,000 Test
retain_loader = DataLoader(Subset(trainset, list(range(0, 4500))), batch_size=64, shuffle=True)
forget_loader = DataLoader(Subset(trainset, list(range(4500, 5000))), batch_size=64, shuffle=True)
test_loader = DataLoader(Subset(testset, list(range(0, 1000))), batch_size=64, shuffle=False)

def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model.to(device)

def train_or_load_baseline_5k():
    path = "baseline_5k.pth"
    model = get_model()
    if os.path.exists(path):
        print("Loading baseline_5k.pth...")
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print("Training baseline model on 5,000 images (15 epochs)...")
        model.train()
        combined_loader = DataLoader(Subset(trainset, list(range(0, 5000))), batch_size=64, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(15):
            for inp, tgt in combined_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                optimizer.zero_grad()
                out = model(inp)
                loss = criterion(out, tgt)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), path)
        print("Saved baseline_5k.pth")
    return model

def evaluate_metrics(model):
    model.eval()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    
    def get_stats(loader):
        losses = []
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion_none(outputs, targets)
                losses.extend(loss.cpu().numpy().tolist())
                _, pred = outputs.max(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
        return np.array(losses), correct / total if total > 0 else 0.0

    retain_losses, retain_acc = get_stats(retain_loader)
    forget_losses, forget_acc = get_stats(forget_loader)
    test_losses, _ = get_stats(test_loader)

    y_true = np.array([1] * len(forget_losses) + [0] * len(test_losses))
    y_scores = np.concatenate([-forget_losses, -test_losses])
    try:
        mia_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        mia_auc = 0.5

    return {
        "retain_accuracy": round(retain_acc * 100, 2),
        "forget_accuracy": round(forget_acc * 100, 2),
        "mia_roc_auc": round(float(mia_auc), 3)
    }

m = train_or_load_baseline_5k()
print("Baseline 5k metrics:", evaluate_metrics(m))
