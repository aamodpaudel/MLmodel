import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt


dataset_path = r"C:\Users\aamod\MLmodel\dataset"
model_save_path = r"C:\Users\aamod\MLmodel\trained_model\dog_emotion_model.pth"
num_epochs = 10
batch_size = 16


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, "val"),
    transform=val_transform
)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


class SimpleDogEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)


model = SimpleDogEmotionModel(num_classes=len(train_dataset.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    
    epoch_loss = running_loss / len(train_dataset)
    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Val Acc: {val_acc:.2f}%")


torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis("off")


images, labels = next(iter(val_loader))
outputs = model(images.to(device))
_, preds = torch.max(outputs, 1)


fig = plt.figure(figsize=(12, 6))
for idx in range(4):
    ax = fig.add_subplot(1, 4, idx+1)
    imshow(images[idx].cpu())
    ax.set_title(f"Pred: {train_dataset.classes[preds[idx]]}\nTrue: {train_dataset.classes[labels[idx]]}")
plt.show()