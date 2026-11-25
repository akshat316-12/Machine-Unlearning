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

def run_lissa_with_anchor(model, damping=0.05, update_lr=0.08, num_steps=8, anchor_lr=0.005):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    # 1. Forget gradient
    forget_grads = [torch.zeros_like(p) for p in model.fc.parameters()]
    total_f = 0
    for inp, tgt in forget_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        model.zero_grad()
        loss = criterion(model(inp), tgt) * inp.size(0)
        grads = torch.autograd.grad(loss, model.fc.parameters())
        for fg, g in zip(forget_grads, grads):
            fg.add_(g.detach())
        total_f += inp.size(0)
    for fg in forget_grads:
        fg.div_(total_f)
        
    v = [fg.clone() for fg in forget_grads]
    u = [vg.clone() for vg in v]
    scale = 1.0 / (damping + 0.1)
    
    # 2. LiSSA HVP
    retain_iter = iter(retain_loader)
    for _ in range(num_steps):
        try:
            r_inp, r_tgt = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            r_inp, r_tgt = next(retain_iter)
            
        r_inp, r_tgt = r_inp.to(device), r_tgt.to(device)
        model.zero_grad()
        loss = criterion(model(r_inp), r_tgt)
        fc_params = list(model.fc.parameters())
        grads = torch.autograd.grad(loss, fc_params, create_graph=True)
        grad_dot_u = sum((g * u_i).sum() for g, u_i in zip(grads, u))
        hvp = torch.autograd.grad(grad_dot_u, fc_params)
        
        with torch.no_grad():
            for i in range(len(u)):
                u[i] = v[i] + (1.0 - scale * damping) * u[i] - scale * hvp[i].detach()
                
    with torch.no_grad():
        for p, delta in zip(model.fc.parameters(), u):
            p.data.add_(update_lr * torch.clamp(delta, -1.0, 1.0))
            
    # 3. Retain projection / anchor (1 fast epoch on retain set)
    model.train()
    optimizer = optim.SGD(model.fc.parameters(), lr=anchor_lr, momentum=0.9)
    for inp, tgt in retain_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inp), tgt)
        loss.backward()
        optimizer.step()
        
    return evaluate_metrics(model)

for d in [0.01, 0.05, 0.1]:
    for ulr in [0.05, 0.1, 0.15]:
        for alr in [0.002, 0.005]:
            m = load_baseline()
            res = run_lissa_with_anchor(m, damping=d, update_lr=ulr, anchor_lr=alr)
            print(f"damping={d}, ulr={ulr}, alr={alr} -> Retain={res['retain_accuracy']}%, Forget={res['forget_accuracy']}%, MIA={res['mia_roc_auc']}")
