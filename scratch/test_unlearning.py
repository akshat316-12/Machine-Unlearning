import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

retain_loader = DataLoader(Subset(trainset, list(range(0, 400))), batch_size=32, shuffle=True)
forget_loader = DataLoader(Subset(trainset, list(range(400, 500))), batch_size=32, shuffle=True)
test_loader = DataLoader(Subset(testset, list(range(0, 200))), batch_size=32, shuffle=False)

def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model.to(device)

def load_baseline():
    model = get_model()
    model.load_state_dict(torch.load("baseline.pth", map_location=device))
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
    test_losses, test_acc = get_stats(test_loader)

    y_true = np.array([1] * len(forget_losses) + [0] * len(test_losses))
    y_scores = np.concatenate([-forget_losses, -test_losses])
    mia_auc = roc_auc_score(y_true, y_scores)

    return {
        "retain_accuracy": round(retain_acc * 100, 2),
        "forget_accuracy": round(forget_acc * 100, 2),
        "mia_roc_auc": round(float(mia_auc), 3),
        "avg_retain_loss": round(float(np.mean(retain_losses)), 3),
        "avg_forget_loss": round(float(np.mean(forget_losses)), 3),
        "avg_test_loss": round(float(np.mean(test_losses)), 3)
    }

# Experiment A: KL-Divergence to Uniform distribution (Maximum Entropy) on Forget Set
def test_uniform_scrubbing(epochs=3, lr=0.005, alpha=1.0):
    model = load_baseline()
    model.train()
    # Update layer4 and fc
    for name, param in model.named_parameters():
        param.requires_grad = ("layer4" in name or "fc" in name)
        
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss()
    
    uniform_target = torch.full((32, 10), 0.1, device=device)
    
    for epoch in range(epochs):
        forget_batches = list(forget_loader)
        f_idx = 0
        for r_inputs, r_targets in retain_loader:
            r_inputs, r_targets = r_inputs.to(device), r_targets.to(device)
            f_inputs, f_targets = forget_batches[f_idx % len(forget_batches)]
            f_idx += 1
            f_inputs, f_targets = f_inputs.to(device), f_targets.to(device)
            
            optimizer.zero_grad()
            r_loss = criterion_ce(model(r_inputs), r_targets)
            
            # Forget: loss to uniform distribution
            f_out = model(f_inputs)
            log_probs = F.log_softmax(f_out, dim=1)
            target = torch.full_like(log_probs, 0.1)
            f_loss = F.kl_div(log_probs, target, reduction='batchmean')
            
            loss = r_loss + alpha * f_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    return evaluate_metrics(model)

# Experiment B: Random Labels scrubbing on Forget Set
def test_random_labels(epochs=3, lr=0.005, alpha=1.0):
    model = load_baseline()
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = ("layer4" in name or "fc" in name)
        
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss()
    
    # Generate fixed random fake labels for forget set
    torch.manual_seed(42)
    
    for epoch in range(epochs):
        forget_batches = list(forget_loader)
        f_idx = 0
        for r_inputs, r_targets in retain_loader:
            r_inputs, r_targets = r_inputs.to(device), r_targets.to(device)
            f_inputs, f_targets = forget_batches[f_idx % len(forget_batches)]
            f_idx += 1
            f_inputs = f_inputs.to(device)
            # Assign random incorrect labels
            fake_targets = torch.randint(0, 10, (f_inputs.size(0),), device=device)
            
            optimizer.zero_grad()
            r_loss = criterion_ce(model(r_inputs), r_targets)
            f_loss = criterion_ce(model(f_inputs), fake_targets)
            
            loss = r_loss + alpha * f_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    return evaluate_metrics(model)

# Experiment C: Pure Gradient Ascent on Forget with Loss Clamping & Retain Anchoring
def test_pure_gradient_ascent(epochs=3, lr=0.001, alpha=0.8, max_loss=2.4):
    model = load_baseline()
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = ("layer4" in name or "fc" in name)
        
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        forget_batches = list(forget_loader)
        f_idx = 0
        for r_inputs, r_targets in retain_loader:
            r_inputs, r_targets = r_inputs.to(device), r_targets.to(device)
            f_inputs, f_targets = forget_batches[f_idx % len(forget_batches)]
            f_idx += 1
            f_inputs, f_targets = f_inputs.to(device), f_targets.to(device)
            
            optimizer.zero_grad()
            r_loss = criterion_ce(model(r_inputs), r_targets)
            f_outputs = model(f_inputs)
            f_ce = criterion_ce(f_outputs, f_targets)
            
            # Maximize forget loss up to max_loss (random guessing level ~ -log(0.1) = 2.302)
            # If loss < max_loss, ascend; otherwise stop ascending
            ascent_loss = -torch.clamp(f_ce, max=max_loss)
            
            loss = r_loss + alpha * ascent_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    return evaluate_metrics(model)

print("Baseline:", evaluate_metrics(load_baseline()))
print("\n--- Testing Uniform Scrubbing ---")
for ep in [2, 3]:
    for lr in [0.003, 0.006]:
        print(f"Uniform ep={ep}, lr={lr}:", test_uniform_scrubbing(epochs=ep, lr=lr, alpha=1.5))

print("\n--- Testing Random Labels ---")
for ep in [2, 3]:
    for lr in [0.003, 0.006]:
        print(f"Random ep={ep}, lr={lr}:", test_random_labels(epochs=ep, lr=lr, alpha=1.0))

print("\n--- Testing Pure Gradient Ascent ---")
for ep in [2, 3]:
    for lr in [0.001, 0.002]:
        for a in [0.8, 1.2]:
            print(f"GA ep={ep}, lr={lr}, a={a}:", test_pure_gradient_ascent(epochs=ep, lr=lr, alpha=a))
