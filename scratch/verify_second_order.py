import sys
sys.path.append('.')
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from engine import run_second_order_lissa_fc, evaluate_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

retain_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(0, 4500)), batch_size=64, shuffle=True)
forget_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(4500, 5000)), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(1000)), batch_size=64, shuffle=False)

model = resnet18(num_classes=10)
model.load_state_dict(torch.load('baseline_5k.pth', map_location=device))
model.to(device)

model = run_second_order_lissa_fc(model, retain_loader, forget_loader, test_loader, device, damping=0.05)
res = evaluate_metrics(model, retain_loader, forget_loader, test_loader, device)
print(f"Second-Order Result -> Retain: {res['retain_accuracy']}%, Forget: {res['forget_accuracy']}%, MIA: {res['mia_roc_auc']}")
