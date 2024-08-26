import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split


class PhysicalModel(nn.Module):
    def __init__(self):
        super(PhysicalModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = torch.from_numpy(np.array(self.data.iloc[:, :2]).astype(np.float32))
        self.y = torch.from_numpy(np.array(self.data.iloc[:, 2]).astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_physical_model():
    data_file_name = 'fitted.csv'
    batch_size = 1
    dataset = MyDataset(data_file_name)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_batch, test_batch = random_split(dataset, [train_size, val_size])
    train_batch = DataLoader(train_batch, batch_size=batch_size, shuffle=True)
    test_batch = DataLoader(test_batch, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PhysicalModel()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_function = nn.MSELoss()
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(train_batch):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            rate = (step + 1) / len(train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{}".format(int(rate * 100), a, b, loss), end="")
        print("\nEpoch {} finished!\n".format(epoch + 1))
    err = 0
    with torch.no_grad():
        for step, data in enumerate(test_batch):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            err += loss_function(outputs, y)
    err /= len(test_batch)
    print(err)
    return model
