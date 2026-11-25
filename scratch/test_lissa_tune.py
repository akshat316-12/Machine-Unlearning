import sys
sys.path.append('.')
import torch
import torch.nn as nn
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

def test_lissa_tuned(damping=0.05, ft_epochs=1, ft_lr=0.01):
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load('baseline_5k.pth', map_location=device))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    update_lr = 0.003
    lissa_steps = 8
    
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
            
    # Stage 2
    if ft_epochs > 0:
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=ft_lr, momentum=0.9, weight_decay=5e-4)
        for epoch in range(ft_epochs):
            forget_list = list(forget_loader)
            f_idx = 0
            for r_inp, r_tgt in retain_loader:
                r_inp, r_tgt = r_inp.to(device), r_tgt.to(device)
                f_inp, f_tgt = forget_list[f_idx % len(forget_list)]
                f_idx += 1
                f_inp, f_tgt = f_inp.to(device), f_tgt.to(device)
                optimizer.zero_grad()
                r_loss = criterion(model(r_inp), r_tgt)
                f_out = model(f_inp)
                true_f_loss = criterion(f_out, f_tgt)
                ascent = -torch.clamp(true_f_loss, max=2.0)
                fake_tgt = (f_tgt + torch.randint(1, 10, f_tgt.shape, device=device)) % 10
                scrub_loss = criterion(f_out, fake_tgt)
                forget_loss = 0.5 * ascent + 0.5 * scrub_loss
                total_loss = r_loss + 0.5 * forget_loss
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
    return evaluate_metrics(model, retain_loader, forget_eval_loader, test_loader, device)

for ft_ep in [0, 1, 2]:
    for ft_lr in [0.005, 0.01]:
        res = test_lissa_tuned(damping=0.05, ft_epochs=ft_ep, ft_lr=ft_lr)
        print(f"ft_ep={ft_ep}, ft_lr={ft_lr} -> Retain: {res['retain_accuracy']}%, Forget: {res['forget_accuracy']}%, MIA: {res['mia_roc_auc']}")
