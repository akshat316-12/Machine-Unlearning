import sys
sys.path.insert(0, ".")
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
        "mia_auc": round(float(mia), 3),
        "f_loss_mean": round(float(np.mean(f_loss)), 3),
        "t_loss_mean": round(float(np.mean(t_loss)), 3)
    }

def test_unlearn(epochs=2, lr=0.015, alpha=0.6, scrub_ratio=0.20, target_scale=0.9):
    model = load_baseline()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion_ce = nn.CrossEntropyLoss()
    torch.manual_seed(42)
    
    for epoch in range(epochs):
        with torch.no_grad():
            t_losses = [criterion_ce(model(inp.to(device)), tgt.to(device)).item() for inp, tgt in test_loader]
            target_cap = float(np.mean(t_losses)) * target_scale
            
        forget_list = list(forget_loader)
        f_idx = 0
        for r_inp, r_tgt in retain_loader:
            r_inp, r_tgt = r_inp.to(device), r_tgt.to(device)
            f_inp, f_tgt = forget_list[f_idx % len(forget_list)]
            f_idx += 1
            f_inp, f_tgt = f_inp.to(device), f_tgt.to(device)
            
            optimizer.zero_grad()
            r_loss = criterion_ce(model(r_inp), r_tgt)
            f_out = model(f_inp)
            true_f_loss = criterion_ce(f_out, f_tgt)
            ascent = -torch.clamp(true_f_loss, max=target_cap)
            fake_tgt = (f_tgt + torch.randint(1, 10, f_tgt.shape, device=device)) % 10
            scrub_loss = criterion_ce(f_out, fake_tgt)
            
            forget_loss = (1.0 - scrub_ratio) * ascent + scrub_ratio * scrub_loss
            total_loss = r_loss + alpha * forget_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    return evaluate(model)

for a in [0.4, 0.5, 0.6]:
    for s_ratio in [0.15, 0.20, 0.25]:
        for scale in [0.85, 0.9, 0.95]:
            res = test_unlearn(epochs=2, lr=0.015, alpha=a, scrub_ratio=s_ratio, target_scale=scale)
            print(f"a={a}, ratio={s_ratio}, scale={scale} -> Retain={res['retain_acc']}%, Forget={res['forget_acc']}%, MIA={res['mia_auc']} (F_loss={res['f_loss_mean']}, T_loss={res['t_loss_mean']})")
