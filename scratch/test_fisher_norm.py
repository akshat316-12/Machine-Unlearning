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
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model.to(device)

def load_baseline():
    model = get_model()
    model.load_state_dict(torch.load("baseline_5k.pth", map_location=device))
    return model

def compute_fisher_diagonal(model, data_loader, device, num_batches=15):
    model.eval()
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    criterion = nn.CrossEntropyLoss()
    count = 0
    for idx, (inputs, targets) in enumerate(data_loader):
        if idx >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += (param.grad.detach() ** 2) * inputs.size(0)
        count += inputs.size(0)
    for name in fisher:
        fisher[name] /= max(count, 1)
        fisher[name] = torch.clamp(fisher[name] / (fisher[name].mean() + 1e-8), 0.0, 10.0)
    return fisher

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

base_model = load_baseline()
fisher = compute_fisher_diagonal(base_model, retain_loader, device)
total_elements = sum(p.numel() for p in base_model.parameters())

def run_fisher_kd_unlearn(epochs=2, lr=0.015, ewc_lambda=2.0, kd_weight=1.0, alpha=0.45, scrub_ratio=0.15):
    student = load_baseline()
    teacher = load_baseline().eval()
    init_params = {name: param.detach().clone() for name, param in teacher.named_parameters()}
    
    student.train()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion_ce = nn.CrossEntropyLoss()
    temp = 2.0
    torch.manual_seed(42)
    
    for epoch in range(epochs):
        with torch.no_grad():
            t_losses = [criterion_ce(student(inp.to(device)), tgt.to(device)).item() for inp, tgt in test_loader]
            target_cap = float(np.mean(t_losses)) * 0.9 if len(t_losses) > 0 else 1.8
            
        forget_list = list(forget_loader)
        f_idx = 0
        for r_inp, r_tgt in retain_loader:
            r_inp, r_tgt = r_inp.to(device), r_tgt.to(device)
            f_inp, f_tgt = forget_list[f_idx % len(forget_list)]
            f_idx += 1
            f_inp, f_tgt = f_inp.to(device), f_tgt.to(device)
            
            optimizer.zero_grad()
            
            r_out = student(r_inp)
            loss_r_ce = criterion_ce(r_out, r_tgt)
            
            # Teacher-Student Distillation on Retain
            with torch.no_grad():
                t_out = teacher(r_inp)
            log_p_s = F.log_softmax(r_out / temp, dim=1)
            p_t = F.softmax(t_out / temp, dim=1)
            loss_kd = F.kl_div(log_p_s, p_t, reduction='batchmean') * (temp ** 2)
            
            # Normalized EWC Penalty
            ewc_sum = 0.0
            for name, param in student.named_parameters():
                if name in fisher:
                    ewc_sum = ewc_sum + (fisher[name] * (param - init_params[name]) ** 2).sum()
            loss_ewc = ewc_sum / total_elements * 1e4
            
            # Forget Alignment
            f_out = student(f_inp)
            true_f_loss = criterion_ce(f_out, f_tgt)
            ascent = -torch.clamp(true_f_loss, max=target_cap)
            fake_tgt = (f_tgt + torch.randint(1, 10, f_tgt.shape, device=device)) % 10
            scrub_loss = criterion_ce(f_out, fake_tgt)
            loss_forget = (1.0 - scrub_ratio) * ascent + scrub_ratio * scrub_loss
            
            total_loss = loss_r_ce + kd_weight * loss_kd + (ewc_lambda * 0.5) * loss_ewc + alpha * loss_forget
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
    return evaluate_metrics(student)

for ewc in [0.5, 1.0, 2.0]:
    for kd in [0.5, 1.0]:
        res = run_fisher_kd_unlearn(epochs=2, lr=0.015, ewc_lambda=ewc, kd_weight=kd)
        print(f"EWC={ewc}, KD={kd} -> Retain={res['retain_accuracy']}%, Forget={res['forget_accuracy']}%, MIA={res['mia_roc_auc']}")
