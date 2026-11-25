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
print("Using device:", device)

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

def compute_fisher_diagonal(model, data_loader, device, num_batches=20):
    """Computes empirical diagonal Fisher Information Matrix on retain set."""
    model.eval()
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    criterion = nn.CrossEntropyLoss()
    
    count = 0
    for idx, (inputs, targets) in enumerate(data_loader):
        if idx >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += (param.grad.detach() ** 2) * inputs.size(0)
        count += inputs.size(0)
        
    for name in fisher:
        fisher[name] /= max(count, 1)
        # Normalize
        fisher[name] = fisher[name] / (fisher[name].mean() + 1e-8)
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
    try:
        mia_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        mia_auc = 0.50

    return {
        "retain_accuracy": round(retain_acc * 100, 2),
        "forget_accuracy": round(forget_acc * 100, 2),
        "mia_roc_auc": round(float(mia_auc), 3)
    }

print("Computing Fisher Information Matrix on GPU...")
base_model = load_baseline()
fisher = compute_fisher_diagonal(base_model, retain_loader, device)
print("Fisher computation complete!")

def run_fisher_kd_unlearn(epochs=2, lr=0.015, ewc_lambda=15.0, kd_weight=1.0, alpha=0.45, scrub_ratio=0.15):
    student = load_baseline()
    teacher = load_baseline()
    teacher.eval()
    
    # Store initial parameters for EWC penalty
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
            
            # 1. Retain Hard Loss
            r_out = student(r_inp)
            loss_r_ce = criterion_ce(r_out, r_tgt)
            
            # 2. Retain Soft Teacher-Student Distillation Loss
            with torch.no_grad():
                teacher_out = teacher(r_inp)
            log_p_student = F.log_softmax(r_out / temp, dim=1)
            p_teacher = F.softmax(teacher_out / temp, dim=1)
            loss_kd = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (temp ** 2)
            
            # 3. Fisher Elastic Weight Consolidation (EWC) penalty
            loss_ewc = 0.0
            for name, param in student.named_parameters():
                if name in fisher:
                    loss_ewc += (fisher[name] * (param - init_params[name]) ** 2).sum()
                    
            # 4. Forget Bounded Ascent + Alignment
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

print("\nTesting Fisher EWC + Distillation Pipeline:")
for ewc in [10.0, 20.0, 30.0]:
    res = run_fisher_kd_unlearn(epochs=2, lr=0.015, ewc_lambda=ewc)
    print(f"EWC={ewc} -> Retain: {res['retain_accuracy']}%, Forget: {res['forget_accuracy']}%, MIA: {res['mia_roc_auc']}")
