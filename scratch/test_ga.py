import torch
import torch.nn as nn
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

# Test what happens when we do pure Gradient Ascent on Forget, followed by repair on Retain:
def test_ga_then_repair(ga_steps=5, ga_lr=0.01, repair_epochs=2, repair_lr=0.005):
    model = load_baseline()
    model.train()
    
    # 1. Gradient Ascent on Forget set (only on fc or layer4+fc)
    optimizer_ga = optim.SGD(model.fc.parameters(), lr=ga_lr)
    criterion = nn.CrossEntropyLoss()
    
    step = 0
    while step < ga_steps:
        for inputs, targets in forget_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_ga.zero_grad()
            outputs = model(inputs)
            loss = -criterion(outputs, targets)
            loss.backward()
            optimizer_ga.step()
            step += 1
            if step >= ga_steps:
                break
                
    m_after_ga = evaluate_metrics(model)
    print(f"After GA ({ga_steps} steps, lr={ga_lr}): Retain={m_after_ga['retain_accuracy']}%, Forget={m_after_ga['forget_accuracy']}%, MIA={m_after_ga['mia_roc_auc']}")
    
    # 2. Repair on Retain set with low lr
    optimizer_repair = optim.SGD(model.fc.parameters(), lr=repair_lr, momentum=0.9)
    for ep in range(repair_epochs):
        for inputs, targets in retain_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_repair.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_repair.step()
            
    m_after_repair = evaluate_metrics(model)
    print(f"After Repair ({repair_epochs} eps): Retain={m_after_repair['retain_accuracy']}%, Forget={m_after_repair['forget_accuracy']}%, MIA={m_after_repair['mia_roc_auc']}")
    return m_after_repair

for steps in [2, 4, 8, 12]:
    for lr in [0.01, 0.02, 0.05]:
        print(f"\n--- GA steps={steps}, lr={lr} ---")
        test_ga_then_repair(ga_steps=steps, ga_lr=lr, repair_epochs=1, repair_lr=0.005)
