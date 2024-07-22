import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
import utils
from model import FaceKeypointResNet50
from tqdm import tqdm
from dataset import FacialKeypointsDataset
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

# model
prune_flag = False
data_aug = True
model_name = ""
training_set = "training"
keypoints = "training_frames_keypoints.csv"

if prune_flag:
    model = FaceKeypointResNet50(pretrained=True, requires_grad=True, pruning_amount=0.2).to(config.DEVICE)
    model_name = "model_prune_0.2_test.pth"
    print("Use prune")
else:
    model = FaceKeypointResNet50(pretrained=True, requires_grad=True).to(config.DEVICE)
    model_name = "model.pth"
    print("Use original")

if data_aug:
    model_name = "model_data_aug.pth"
    training_set = "training_aug"
    keypoints = "rotated_keypoints.csv"

print("successfully defined model")

# optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LR)
criterion = nn.SmoothL1Loss()

train_data=FacialKeypointsDataset(f'{config.ROOT_PATH}/{keypoints}',f'{config.ROOT_PATH}/{training_set}')
valid_data=FacialKeypointsDataset(f'{config.ROOT_PATH}/test_frames_keypoints.csv',f'{config.ROOT_PATH}/test')
train_loader=DataLoader(train_data,batch_size=config.BATCH_SIZE,shuffle=True)
valid_loader=DataLoader(valid_data,batch_size=config.BATCH_SIZE,shuffle=False)

# training function
def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss


def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
        
    valid_loss = valid_running_loss/counter
    return valid_loss

train_loss = []
val_loss = []
for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    train_epoch_loss = fit(model, train_loader, train_data)
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.show()
torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'train_loss':train_loss,
            'val_loss':val_loss,
            }, f"{config.OUTPUT_PATH}/{model_name}")
print('DONE TRAINING')