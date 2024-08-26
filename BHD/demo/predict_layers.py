import os

import torch
import csv

from physical.model import get_physical_model

os.chdir('../physical')
model = get_physical_model()
os.chdir('../demo')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

items = []
file_name = '202307031749.csv'
with open('attack_data/' + file_name, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) == 8:
            items.append(row)
data = []
for item in items[1:]:
    gamma1, gamma2, d1, d2 = float(item[-4]), float(item[-3]), float(item[-2]), float(item[-1])
    error1 = 1e10
    error2 = 1e10
    best_layer1 = 0
    best_layer2 = 0
    for layers in range(20):
        output1 = model(torch.tensor([float(layers), float(d1)]).to(device))
        output2 = model(torch.tensor([float(layers), float(d2)]).to(device))
        if abs(output1 - gamma1) < error1:
            error1 = abs(output1 - gamma1)
            best_layer1 = layers
        if abs(output2 - gamma2) < error2:
            error2 = abs(output2 - gamma2)
            best_layer2 = layers
    data.append(item + [best_layer1, best_layer2])
print(data)
with open('attack_data/fit' + file_name, "w", newline="", encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['AP1', 'AP2', 'x', 'y', 'gamma1', 'gamma2', 'd1', 'd2', 'layers1', 'layers2'])
    writer.writerows(data)
