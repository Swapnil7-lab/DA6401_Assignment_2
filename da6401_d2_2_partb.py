import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import wandb




wandb.login(key="3d199b9bde866b3494cda2f8bb7c7a633c9fdade") 
wandb.init(project="DA6401_Assignment_2")  

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data(bs):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    home_path = "/content/inaturalist_12K"
    train_path = os.path.join(home_path, 'train')
    test_path = os.path.join(home_path, 'val')

    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, bs, shuffle=True)
    val_loader = DataLoader(val_dataset, bs, shuffle=False)
    test_loader = DataLoader(test_dataset, bs, shuffle=False)

    root = pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

    return train_loader, val_loader, test_loader, classes

train_loader, val_loader, test_loader, classes = load_data(4)

# Load pre-trained ResNet50 model
model = torchvision.models.resnet50(pretrained=True)

# Freeze all layers except the last fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(5):
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 2000 == 1999:
            print(f'Epoch [{epoch+1}, {batch_idx+1}] loss: {running_loss/2000:.3f}')
            running_loss = 0.0

    # Log metrics to WandB
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            test_acc += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100 * test_acc / len(test_loader.dataset)
    wandb.log({
        'epoch': epoch+1,
        'test_loss': test_loss,
        'test_acc': test_acc
    })

print('Training Completed')

# Fine-tuning strategies
# 1. Fine-tune only the last few layers
# 2. Fine-tune some layers and retrain others
# 3. Fine-tune all the layers

# Example of fine-tuning some layers
# for name, param in model.named_parameters():
#     if "layer4" in name:
#         param.requires_grad = True

# optimizer = optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.001, momentum=0.9)

# WandB Finish
wandb.finish()
