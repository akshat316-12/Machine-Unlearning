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
    test_losses, _ = get_stats(test_loader)

    y_true = np.array([1] * len(forget_losses) + [0] * len(test_losses))
    y_scores = np.concatenate([-forget_losses, -test_losses])
    mia_auc = roc_auc_score(y_true, y_scores)

    return {
        "retain_accuracy": round(retain_acc * 100, 2),
        "forget_accuracy": round(forget_acc * 100, 2),
        "mia_roc_auc": round(float(mia_auc), 3)
    }

def test_lissa_target_shift(damping=0.05, update_lr=0.05, project_lr=0.015):
    model = load_baseline()
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    torch.manual_seed(42)
    # 1. Compute difference gradient: grad(true_label) - grad(fake_label)
    diff_grads = [torch.zeros_like(p) for p in model.fc.parameters()]
    total_f = 0
    for inp, tgt in forget_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        fake_tgt = (tgt + torch.randint(1, 10, tgt.shape, device=device)) % 10
        model.zero_grad()
        out = model(inp)
        l_true = criterion(out, tgt) * inp.size(0)
        l_fake = criterion(out, fake_tgt) * inp.size(0)
        loss = l_true - 0.5 * l_fake
        grads = torch.autograd.grad(loss, model.fc.parameters())
        for dg, g in zip(diff_grads, grads):
            dg.add_(g.detach())
        total_f += inp.size(0)
    for dg in diff_grads:
        dg.div_(max(total_f, 1))
        
    v = [dg.clone() for dg in diff_grads]
    u = [vg.clone() for vg in v]
    scale = 1.0 / (damping + 0.1)
    
    # 2. LiSSA HVP
    retain_iter = iter(retain_loader)
    for _ in range(8):
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
                
    # 3. Apply influence step
    with torch.no_grad():
        for p, delta in zip(model.fc.parameters(), u):
            p.data.add_(update_lr * torch.clamp(delta, -1.0, 1.0))
            
    # 4. Retain projection
    model.train()
    optimizer = optim.SGD(model.fc.parameters(), lr=project_lr, momentum=0.9)
    for inp, tgt in retain_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inp), tgt)
        loss.backward()
        optimizer.step()
        
    return evaluate_metrics(model)

for ulr in [0.03, 0.06, 0.09]:
    for plr in [0.005, 0.01]:
        m = test_lissa_target_shift(damping=0.05, update_lr=ulr, project_lr=plr)
        print(f"ulr={ulr}, plr={plr} -> Retain={m['retain_accuracy']}%, Forget={m['forget_accuracy']}%, MIA={m['mia_roc_auc']}")
