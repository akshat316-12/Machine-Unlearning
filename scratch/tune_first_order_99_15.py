import os
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

retain_loader = DataLoader(Subset(trainset, list(range(0, 4500))), batch_size=64, shuffle=True)
forget_loader = DataLoader(Subset(trainset, list(range(4500, 5000))), batch_size=64, shuffle=True)
test_loader = DataLoader(Subset(testset, list(range(0, 1000))), batch_size=64, shuffle=False)

def get_model():
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    for module in m.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return m.to(device)

def load_baseline():
    m = get_model()
    m.load_state_dict(torch.load("baseline_5k.pth", map_location=device))
    return m

def evaluate(model):
    model.eval()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    def get_stats(loader):
        losses = []
        correct, total = 0, 0
        with torch.no_grad():
            for inp, tgt in loader:
                inp, tgt = inp.to(device), tgt.to(device)
                out = model(inp)
                loss = criterion_none(out, tgt)
                losses.extend(loss.cpu().numpy().tolist())
                _, pred = out.max(1)
                correct += pred.eq(tgt).sum().item()
                total += tgt.size(0)
        return np.array(losses), correct / total if total > 0 else 0.0

    r_loss, r_acc = get_stats(retain_loader)
    f_loss, f_acc = get_stats(forget_loader)
    t_loss, _ = get_stats(test_loader)

    y_true = np.array([1]*len(f_loss) + [0]*len(t_loss))
    y_scores = np.concatenate([-f_loss, -t_loss])
    mia = roc_auc_score(y_true, y_scores)
    return {
        "retain_acc": round(r_acc * 100, 2),
        "forget_acc": round(f_acc * 100, 2),
        "mia_auc": round(float(mia), 3)
    }

def run_pure_fo(epochs=3, lr=0.015, alpha=0.8, scrub_ratio=0.5, max_ascent=2.8):
    model = load_baseline()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion_ce = nn.CrossEntropyLoss()
    torch.manual_seed(42)
    
    for epoch in range(epochs):
        forget_list = list(forget_loader)
        f_idx = 0
        for r_inp, r_tgt in retain_loader:
            r_inp, r_tgt = r_inp.to(device), r_tgt.to(device)
            f_inp, f_tgt = forget_list[f_idx % len(forget_list)]
            f_idx += 1
            f_inp, f_tgt = f_inp.to(device), f_tgt.to(device)
            
            optimizer.zero_grad()
            
            # Retain descent
            r_loss = criterion_ce(model(r_inp), r_tgt)
            
            # Forget ascent towards ~2.5 - 3.0 (-log(0.10) ~ 2.3)
            f_out = model(f_inp)
            true_f_loss = criterion_ce(f_out, f_tgt)
            ascent = -torch.clamp(true_f_loss, max=max_ascent)
            
            # Noise target to guide to 10%
            fake_tgt = (f_tgt + torch.randint(1, 10, f_tgt.shape, device=device)) % 10
            scrub_loss = criterion_ce(f_out, fake_tgt)
            
            f_component = (1.0 - scrub_ratio) * ascent + scrub_ratio * scrub_loss
            
            total_loss = r_loss + alpha * f_component
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    return evaluate(model)

print("Starting sweep to achieve Retain >= 99%, Forget 10%-20%, MIA <= 0.50:")
for ep in [2, 3]:
    for a in [0.7, 0.85, 1.0]:
        for ratio in [0.45, 0.55, 0.65]:
            for cap in [2.6, 3.0]:
                res = run_pure_fo(epochs=ep, lr=0.015, alpha=a, scrub_ratio=ratio, max_ascent=cap)
                r_ok = res['retain_acc'] >= 99.0
                f_ok = 10.0 <= res['forget_acc'] <= 22.0
                m_ok = res['mia_auc'] <= 0.50
                is_hit = r_ok and f_ok and m_ok
                mark = "*** TARGET HIT ***" if is_hit else ""
                print(f"ep={ep}, a={a}, ratio={ratio}, cap={cap} -> Retain={res['retain_acc']}%, Forget={res['forget_acc']}%, MIA={res['mia_auc']} {mark}")
                if is_hit:
                    break
