import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model.resnet import ResNet18
import time


def train_model(model, loader, criterion, optimizer, device):
    print("training started")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward Pass 
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy


def evaluate(model, loader, criterion, device):
    print("Evaluation is starting")
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy            



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_path = "C:/Deep Learning/Car_Damage_Project/training/dataset"

    dataset = datasets.ImageFolder(root=dataset_path, transform=image_transforms)
    len(dataset)
    print("Classes", dataset.classes)
    print("dataset.class idx", dataset.class_to_idx)
    print("Len of classes", len(dataset.classes))
    
    train_size = int(0.80*len(dataset))
    val_size = len(dataset) - train_size

    train_size , val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)
    
    print("Training is Starting")
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(1, 13):  # 10 epochs
        start = time.time()
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    
    
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f} | Time: {time.time()-start:.2f}s")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_resnet18.pth")
        print("✅ Model saved!")

if __name__ == "__main__":
    main()
