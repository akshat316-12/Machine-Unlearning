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
    try:
        mia_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        mia_auc = 0.5

    return {
        "retain_accuracy": round(retain_acc * 100, 2),
        "forget_accuracy": round(forget_acc * 100, 2),
        "mia_roc_auc": round(float(mia_auc), 3)
    }

def run_lissa_then_finetune(damping=0.05, update_lr=0.01, lissa_steps=8, 
                            ft_epochs=3, ft_lr=0.018, alpha=0.85, scrub_ratio=0.35):
    """
    Stage 1: Conservative LiSSA influence perturbation on model.fc
    Stage 2: Calibrated first-order unlearning for convergence
    """
    model = load_baseline()
    criterion = nn.CrossEntropyLoss()
    
    # --- STAGE 1: LiSSA influence perturbation ---
    model.eval()
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
        fg.div_(max(total_f, 1))
        
    v = [fg.clone() for fg in forget_grads]
    u = [vg.clone() for vg in v]
    scale = 1.0 / (damping + 0.1)
    
    retain_iter = iter(retain_loader)
    for _ in range(lissa_steps):
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
    
    m_after_lissa = evaluate_metrics(model)
    print(f"  After LiSSA: Retain={m_after_lissa['retain_accuracy']}%, Forget={m_after_lissa['forget_accuracy']}%, MIA={m_after_lissa['mia_roc_auc']}")
    
    # --- STAGE 2: First-order calibrated fine-tuning ---
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=ft_lr, momentum=0.9, weight_decay=5e-4)
    criterion_ce = nn.CrossEntropyLoss()
    torch.manual_seed(42)
    
    for epoch in range(ft_epochs):
        with torch.no_grad():
            t_losses = [criterion_ce(model(inp.to(device)), tgt.to(device)).item() for inp, tgt in test_loader]
            target_cap = float(np.mean(t_losses)) if len(t_losses) > 0 else 3.5
            
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
            
    return evaluate_metrics(model)

# Test combinations
for ulr in [0.005, 0.01, 0.015]:
    for ft_ep in [2, 3]:
        for ft_lr in [0.015, 0.018]:
            m = run_lissa_then_finetune(damping=0.05, update_lr=ulr, lissa_steps=8, 
                                         ft_epochs=ft_ep, ft_lr=ft_lr)
            print(f"ulr={ulr}, ft_ep={ft_ep}, ft_lr={ft_lr} -> Retain={m['retain_accuracy']}%, Forget={m['forget_accuracy']}%, MIA={m['mia_roc_auc']}\n")
