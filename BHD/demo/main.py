import csv
import datetime
import math

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import random_split, DataLoader

from ResNet.model import ResNet
from atttack import Attack
from dataset import one_hot_encode, MyDataset, get_coordinate, get_error, get_error_list

torch.manual_seed(3407)

batch_size = 32
train_file_path = 'data/data.csv'
epoch_num = 50
adv_c = 0.1
adv_kappa = 2
adv_mode = 'L2'
adv_increase = False
adv_bias = 0.5
save_path = './weight_my.pth'

dic, inverse_dic = one_hot_encode(file_path=train_file_path)

dataset = MyDataset(file_path=train_file_path, dic=dic)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
test_size, val_size = int(test_size / 2), test_size - int(test_size / 2)
test_dataset, val_dataset = random_split(test_dataset, [test_size, val_size])
train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_batch = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet(in_channels=1, classes=len(dic))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
loss_function = nn.CrossEntropyLoss()
best_error = 1e9

print("Batch size: {}\nepoch num: {}\n".format(batch_size, epoch_num))
print("Train batch num: {}\nVal batch num:   {}\nTest batch num:  {}".format(
    len(train_batch), len(val_batch), len(test_batch))
)

for epoch in range(epoch_num):
    # train
    model.train()
    running_loss = 0.0
    for step, data in enumerate(train_batch):
        x, y, site = data
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
    print("")
    model.eval()
    err = 0.0
    with torch.no_grad():
        for step, data in enumerate(val_batch):
            x, y, site = data
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            outputs = model(x)
            err += get_error(outputs.cpu(), y.cpu(), inverse_dic)
        err /= len(val_batch)
        if best_error > err:
            best_error = err
            torch.save(model.state_dict(), save_path)
            print('current weight saved!')
        print('[epoch %d] validate err: %f m\n' % (epoch + 1, err))


def get_demo_error(x, y):
    idx = dic[y]
    y = torch.zeros(len(dic))
    y[idx - 1] = 1
    return get_error(x, y.view(1, -1), inverse_dic)


# print(get_demo_error(model(torch.tensor([-34.2166666666666, -47]).view(n=3, n=3, 2).to(device)).cpu(), (2.5, n=3.9)))
# print(get_demo_error(model(torch.tensor([-47.8833333333333, -33]).view(n=3, n=3, 2).to(device)).cpu(), (4, 9.n=3)))
# print(get_demo_error(model(torch.tensor([-42.95, -48]).view(n=3, n=3, 2).to(device)).cpu(), (-0.8, 2.8)))
# print(get_demo_error(model(torch.tensor([-33.1833333333333, -47]).view(n=3, n=3, 2).to(device)).cpu(), (2.5, 2.8)))
# print(get_demo_error(model(torch.tensor([-43.1833333333333, -55]).view(n=3, n=3, 2).to(device)).cpu(), (2.5, 4.6)))
# print(get_demo_error(model(torch.tensor([-39.0833333333333, -44]).view(n=3, n=3, 2).to(device)).cpu(), (-0.8, n=3.9)))
# print(get_demo_error(model(torch.tensor([-50, -39.4333333333333]).view(n=3, n=3, 2).to(device)).cpu(), (3.5, 7.3)))
# print(get_demo_error(model(torch.tensor([-56, -34.2833333333333]).view(n=3, n=3, 2).to(device)).cpu(), (2.5, 9.n=3)))
# print(get_demo_error(model(torch.tensor([-43, -37.9333333333333]).view(n=3, n=3, 2).to(device)).cpu(), (0.5, 4.6)))
# print(get_demo_error(model(torch.tensor([-35.05, -45]).view(n=3, n=3, 2).to(device)).cpu(), (n=3.5, n=3.9)))
# print(get_demo_error(model(torch.tensor([-48, -38.3166666666666]).view(n=3, n=3, 2).to(device)).cpu(), (2, 8.2)))
# print(get_demo_error(model(torch.tensor([-45.0833333333333, -55]).view(n=3, n=3, 2).to(device)).cpu(), (2.5, 3.7)))
# print(get_demo_error(model(torch.tensor([-50.05, -52]).view(n=3, n=3, 2).to(device)).cpu(), (4.75, 4.6)))
# print(get_demo_error(model(torch.tensor([-44, -39.95]).view(n=3, n=3, 2).to(device)).cpu(), (0.5, 9.n=3)))
# print(get_demo_error(model(torch.tensor([-51, -48.5166666666666]).view(n=3, n=3, 2).to(device)).cpu(), (n=3, 3.7)))

# exit(0)

rss = []
x = []
y = []

# test
err_normal = 0.0
err_adv = 0.0
bias = 0.0
partial_bias = 0.0
attack = Attack(model, device, c=adv_c, kappa=adv_kappa, mode=adv_mode, isIncrease=adv_increase, bias=adv_bias)
model.load_state_dict(torch.load(save_path))
t = datetime.datetime.now().strftime("%Y%m%d%H%M")
with open('attack_data/' + t + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['X', 'Y', 'AP1', 'AP2', 'error', 'AP1_attack', 'AP2_attack', 'error_adv', 'layers1', 'layers2'])
    for step, data in enumerate(test_batch):
        x, y, site = data
        x = x.view(-1, 1, len(x[0])).to(device)
        y = y.view(-1, len(y[0]))

        outputs = model(x).cpu()
        err_normal += get_error(outputs, y, inverse_dic)

        x, y, site = data

        # write test data and result
        x_adv, y_adv, layers1, layers2 = attack.generate_data(data)
        x_adv1 = x_adv.view(-1, 1, len(x[0])).to(device)
        y_adv1 = y_adv.view(-1, len(y[0]))
        bias += torch.abs(torch.abs(x_adv.cpu() - x) / x).mean()
        abs_diff = torch.abs(x_adv.cpu() - x)
        partial_bias += torch.abs((abs_diff / x))[abs_diff != 0].mean()
        outputs_adv = model(x_adv1).cpu()
        err_adv += get_error(outputs_adv, y_adv1, inverse_dic)

        error_list = get_error_list(outputs.detach(), y, inverse_dic)
        error_adv_list = get_error_list(outputs_adv.detach(), y_adv1, inverse_dic)
        for i in range(len(x)):
            temp = inverse_dic[int(np.argmax(y[i])) + 1]
            x_abs, y_abs = temp[0], temp[1]
            gammas = []
            writer.writerow([x_abs, y_abs] + x.tolist()[i] + [error_list[i]] + x_adv.tolist()[i] + [error_adv_list[i]]
                            + [layers1[i], layers2[i]])

print("Normal prediction's error is {} m".format(err_normal / len(test_batch)))
print("After attack, prediction's Error is {} m".format(err_adv / len(test_batch)))
print("Attack bias: {}%".format(bias / len(test_batch) * 100))
print("Partial attack bias: {}%".format(partial_bias / len(test_batch) * 100))
