import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from engine import evaluate_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

retain_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(0, 4500)), batch_size=64, shuffle=True)
forget_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(4500, 5000)), batch_size=64, shuffle=True)
forget_eval_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(4500, 5000)), batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(1000)), batch_size=64, shuffle=False)

def run_uniform_unlearning(epochs=2, lr=0.01, beta=0.5):
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load('baseline_5k.pth', map_location=device))
    model.to(device)
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion_ce = nn.CrossEntropyLoss()
    
    uniform_targets = torch.full((64, 10), 0.1, device=device)
    
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
            r_out = model(r_inp)
            r_loss = criterion_ce(r_out, r_tgt)
            
            # Uniform dispersion on forget (KL divergence to uniform distribution)
            f_out = model(f_inp)
            log_probs = F.log_softmax(f_out, dim=1)
            u_tgt = torch.full_like(log_probs, 0.1)
            # KL(U || p) = sum(U * (log U - log p)) -> min -sum(log p)/10
            # Which is cross entropy with soft uniform distribution!
            f_loss = -(log_probs.mean(dim=1)).mean()
            
            total_loss = r_loss + beta * f_loss
            total_loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    metrics = evaluate_metrics(model, retain_loader, forget_eval_loader, test_loader, device)
    return metrics

for ep in [1, 2]:
    for lr in [0.005, 0.01, 0.015]:
        for beta in [0.2, 0.4, 0.6]:
            res = run_uniform_unlearning(epochs=ep, lr=lr, beta=beta)
            print(f"ep={ep}, lr={lr}, beta={beta} -> Retain: {res['retain_accuracy']}%, Forget: {res['forget_accuracy']}%, MIA: {res['mia_roc_auc']}")
