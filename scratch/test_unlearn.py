import sys
sys.path.append('.')
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from engine import run_first_order_unlearning, evaluate_metrics

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

configs = [
    (2, 0.015, 0.7, 0.45, 2.6),
    (3, 0.015, 0.7, 0.45, 2.6),
    (4, 0.018, 0.7, 0.45, 2.6),
]

for epochs, lr, alpha, scrub, max_asc in configs:
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load('baseline_5k.pth', map_location=device))
    model.to(device)
    model = run_first_order_unlearning(model, retain_loader, forget_loader, test_loader, device, epochs=epochs, lr=lr, alpha=alpha, scrub_ratio=scrub, max_ascent=max_asc)
    metrics = evaluate_metrics(model, retain_loader, forget_eval_loader, test_loader, device)
    print(f"ep={epochs}, lr={lr}, alpha={alpha}, scrub={scrub}, max_asc={max_asc} -> Retain: {metrics['retain_accuracy']}%, Forget: {metrics['forget_accuracy']}%, MIA: {metrics['mia_roc_auc']}")
