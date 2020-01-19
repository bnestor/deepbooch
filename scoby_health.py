"""
scoby_health.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import glob
from tqdm import tqdm
import random

import numpy as np
from sklearn.metrics import roc_auc_score


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 3 )
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.conv4 = nn.Conv2d(16, 5, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4205, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class kombuchaDataset(Dataset):
    def __init__(self, healthy, unhealthy):
        """
        """
        self.imgs=healthy+unhealthy
        self.labels=[1]*len(healthy)+[0]*len(unhealthy)
        self.transform = transforms.Compose([transforms.Resize((500,500), interpolation=2),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor()])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, item):

        image = Image.open(self.imgs[item])
        image = self.transform(image)

        return image, torch.tensor(self.labels[item])




def train():
    device='cpu'
    net = Net()
    #dataset
    healthy=glob.glob('data/normal/*.jpg')
    unhealthy=glob.glob('data/sick/*.jpg')

    random.shuffle(healthy)
    random.shuffle(unhealthy)
    healthy_train, healthy_test = healthy[:int(0.7*len(healthy))], healthy[int(0.7*len(healthy)):]
    unhealthy_train, unhealthy_test = unhealthy[:int(0.7*len(unhealthy))], unhealthy[int(0.7*len(unhealthy)):]

    print(len(healthy_train), len(healthy_test), len(healthy))
    print(len(unhealthy_train), len(unhealthy_test), len(unhealthy))

    dataset = kombuchaDataset(healthy_train, unhealthy_train)
    train_dataloader = DataLoader(dataset, shuffle = True, batch_size=2, drop_last=False)

    net.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    epochs = tqdm(range(100), leave=False)
    for epoch in epochs:
        batches= tqdm(train_dataloader, total=len(train_dataloader))
        for batch in batches:

            img, label =batch
            pred = net(img.to(device).float())

            loss= criterion(pred.view(-1), label.float().to(device).view(-1))
            stable_loss = loss.cpu().data.numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batches.set_description(desc=f'{stable_loss}')


    testdataset = kombuchaDataset(healthy_test, unhealthy_test)
    test_dataloader = DataLoader(testdataset, shuffle = True, batch_size=1, drop_last=False)
    labels=[]
    predictions=[]
    m=nn.Sigmoid()
    net.eval()
    batches= tqdm(test_dataloader, total=len(test_dataloader))
    for batch in batches:
        img, label =batch
        with torch.no_grad():
            pred = net(img.to(device).float())

        predictions.append(m(pred).cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    predictions= np.asarray(predictions).ravel()
    labels=np.asarray(labels).ravel()

    print(predictions)

    print(labels)

    print(roc_auc_score(labels, predictions))




if __name__ == "__main__":
    train()
