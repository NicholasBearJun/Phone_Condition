import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import copy
import pathlib
import requests
import zipfile
import os


#Dataset Directory
data_dir = pathlib.Path("C:/.../Broken Phone Machine Vision/data/Good_N_Broken/DA")
train_dir = os.path.join(data_dir, "Train")
val_dir = os.path.join(data_dir, "Validation")
print(train_dir)
print(val_dir)

# Image transformation
prep_img_mean = [0.485, 0.456, 0.406]
prep_img_std = [0.229, 0.224, 0.225]

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=prep_img_mean, std=prep_img_std),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=prep_img_mean, std=prep_img_std),
    ]),
}

# Get dataset
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms["val"])

# Dateloader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

class_names = train_dataset.classes

# Set dictionary
dataloaders = {
    "train": train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train": len(train_dataset),
    "val": len(val_dataset)
}

# try Cuba before CPU
device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)

print(f"Classes: {class_names}")
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

#Model Training
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train() 
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    #optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f"{phase}\tloss: {epoch_loss:.3f}, accuracy: {epoch_acc:.3f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training completed in "
        f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Best val accuracy: {best_accuracy:.3f}")

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

# Use Pre-trained EfficientNet-B0
model_conv = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')

# Freeze all layers
for param in model_conv.parameters():
    param.requires_grad = False

# Modify classifier layer
num_features = model_conv.classifier[1].in_features
model_conv.classifier[1] = nn.Linear(num_features, len(class_names))

#model to Cuba or PCU
model_conv = model_conv.to(device)



# loss function
criterion = nn.CrossEntropyLoss()

optimizer_conv = torch.optim.AdamW(model_conv.classifier[1].parameters(), lr=1e-4, weight_decay=1e-3)

exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_conv, T_max=30)

#Train and Evalaute
if __name__ == '__main__':
    model_conv = train_model(
        model=model_conv,
        criterion=criterion,
        optimizer=optimizer_conv,
        scheduler=exp_lr_scheduler,
        num_epochs=6,
    )

torch.save(model_conv, "model_phone_class.pt")
print("Model saved as model_phone_class.pt")

print("Ran with no prob")
