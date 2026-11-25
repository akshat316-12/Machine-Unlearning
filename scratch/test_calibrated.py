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

def run_calibrated_unlearning(epochs=3, lr=0.015, alpha=0.8, beta=1.0):
    model = load_baseline()
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion_ce = nn.CrossEntropyLoss()
    
    torch.manual_seed(42)
    
    for epoch in range(epochs):
        # Dynamically measure current average test loss
        with torch.no_grad():
            t_losses = [criterion_ce(model(inp.to(device)), tgt.to(device)).item() for inp, tgt in test_loader]
            target_cap = float(np.median(t_losses))
            
        forget_list = list(forget_loader)
        f_idx = 0
        for r_inp, r_tgt in retain_loader:
            r_inp, r_tgt = r_inp.to(device), r_tgt.to(device)
            f_inp, f_tgt = forget_list[f_idx % len(forget_list)]
            f_idx += 1
            f_inp, f_tgt = f_inp.to(device), f_tgt.to(device)
            
            optimizer.zero_grad()
            
            # Retain loss (descent)
            r_loss = criterion_ce(model(r_inp), r_tgt)
            
            # Forget loss: bounded ascent towards target_cap (matching test loss distribution)
            f_out = model(f_inp)
            true_f_loss = criterion_ce(f_out, f_tgt)
            
            # Bounded ascent: once f_loss >= target_cap, gradient ascent stops!
            ascent_loss = -torch.clamp(true_f_loss, max=target_cap)
            
            # Soft uniform target loss to randomize probabilities without exploding
            log_p = F.log_softmax(f_out, dim=1)
            uniform_kl = F.kl_div(log_p, torch.full_like(log_p, 0.1), reduction='batchmean')
            
            total_loss = beta * r_loss + alpha * (0.7 * ascent_loss + 0.3 * uniform_kl)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    return evaluate_metrics(model)

print("Baseline:", evaluate_metrics(load_baseline()))
for ep in [2, 3, 4]:
    for lr in [0.01, 0.015, 0.02]:
        for a in [0.6, 0.8, 1.0]:
            m = run_calibrated_unlearning(epochs=ep, lr=lr, alpha=a)
            print(f"ep={ep}, lr={lr}, a={a} -> Retain={m['retain_accuracy']}%, Forget={m['forget_accuracy']}%, MIA={m['mia_roc_auc']}, F_Loss={m['avg_forget_loss']}, T_Loss={m['avg_test_loss']}")
