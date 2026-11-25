import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, "(", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", ")")

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

# 5,000 images baseline training
combined_loader = DataLoader(Subset(trainset, list(range(0, 5000))), batch_size=64, shuffle=True)

model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
for m in model.modules():
    if isinstance(m, nn.ReLU):
        m.inplace = False
model = model.to(device)

print("Training baseline ResNet-18 on 5,000 CIFAR-10 images for 25 epochs on GPU...")
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(1, 26):
    correct, total, epoch_loss = 0, 0, 0.0
    for inputs, targets in combined_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    scheduler.step()
    if epoch % 5 == 0 or epoch == 25:
        print(f"Epoch {epoch}/25 - Loss: {epoch_loss/total:.4f} - Accuracy: {100.0*correct/total:.2f}%")

torch.save(model.state_dict(), "baseline_5k.pth")
print("Successfully saved 5,000-sample model weights to baseline_5k.pth!")
