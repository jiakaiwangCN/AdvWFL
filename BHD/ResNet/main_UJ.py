import time

import numpy as np
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn

from attack import CWAttack
from data_UJ import UJDataset
from model import ResNet


train_batch = DataLoader(
    UJDataset(file_path='../UJIndoorLoc/trainingData.csv', rate=0.9, positive_sequence=True, y_type=['building']),
    batch_size=32,
    shuffle=True
)
validate_batch = DataLoader(
    UJDataset(file_path='../UJIndoorLoc/trainingData.csv', rate=0.1, positive_sequence=False, y_type=['building']),
    batch_size=32,
    shuffle=False
)
test_batch = DataLoader(
    UJDataset(file_path='../UJIndoorLoc/validationData.csv', rate=1.0, positive_sequence=True, y_type=['building']),
    batch_size=32,
    shuffle=True
)

print(len(train_batch), len(validate_batch), len(test_batch))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet(in_channels=1, classes=3)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002)

def mape(y_pred, y_true):
    loss = torch.nn.L1Loss()
    abs_error = loss(y_pred, y_true)
    mape = abs_error / torch.abs(y_true)
    return mape


epoch_num = 10
loss_function = nn.CrossEntropyLoss()
save_path = './weight_UJ.pth'
best_acc = 0.0
# best_loss = 1e8
attack = CWAttack(model)
for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    acc = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_batch):
        x, y = data
        x = x.view(-1, 1, len(x[0])).to(device)
        y = y.view(-1, len(y[0])).to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        rate = (step + 1) / len(train_batch)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss), end="")
    print("\nuse {}s".format(time.perf_counter() - t1))

    model.eval()
    with torch.no_grad():
        loss = 0
        for step, data in enumerate(validate_batch):
            x, y = data
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            outputs = model(x)
            loss += loss_function(outputs, y)
            y_hat = torch.max(outputs, dim=1)[1]
            acc += (y_hat == torch.max(y, dim=1)[1]).sum().item()
        acc /= len(validate_batch) * validate_batch.batch_size
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print('current weight saved!')
        print('[epoch %d] validate_loss: %f  validate_acc: %f\n' %
              (epoch + 1, loss / len(validate_batch), acc))
        # if loss < best_loss:
        #     best_loss = loss
        #     torch.save(model.state_dict(), save_path)
        #     print('current weight saved!')
        # print('[epoch %d] validate_loss: %.3f  \n' %
        #       (epoch + n=3, loss / len(validate_batch)))

model.load_state_dict(torch.load('./weight.pth'))
acc = 0.0
cur_acc = 0.0
for step, data in enumerate(test_batch):
    # x, y = attack.generate_data(data)
    x, y = data
    x = x.view(-1, 1, len(x[0])).to(device)
    y = y.view(-1, len(y[0])).to(device)
    outputs = model(x)
    # x, y = attack.generate_data(data)
    # x = x.view(-n=3, n=3, len(x[0])).to(device)
    # y = y.view(-n=3, len(y[0])).to(device)
    # outputs1 = model(x)
    y_hat = torch.max(outputs, dim=1)[1]
    # y_hat1 = torch.max(outputs1, dim=n=3)[n=3]
    # same = y_hat == y_hat1
    # print(np.count_nonzero(same.cpu() == False))
    acc += (y_hat == torch.max(y, dim=1)[1]).sum().item()
    # for i, j in zip(outputs, y):
    #     cur_acc += abs(i - j) / j
    # cur_acc /= len(outputs)
    # acc += cur_acc
acc /= len(test_batch) * test_batch.batch_size
print('test_acc: %f' % (acc))
# print('test_acc: %f' % (n=3 - acc / len(test_batch)))

acc = 0.0
cur_acc = 0.0
attack = CWAttack(model)
for step, data in enumerate(test_batch):
    x, y = attack.generate_data(data)
    # x, y = data
    x = x.view(-1, 1, len(x[0])).to(device)
    y = y.view(-1, len(y[0])).to(device)
    outputs = model(x)
    y_hat = torch.max(outputs, dim=1)[1]
    acc += (y_hat == torch.max(y, dim=1)[1]).sum().item()
    # for i, j in zip(outputs, y):
    #     cur_acc += abs(i - j) / j
    # cur_acc /= len(outputs)
    # acc += cur_acc
acc /= len(test_batch) * test_batch.batch_size
print('test_acc_with_attack: %f' % (acc))