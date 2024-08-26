import csv
import datetime
import time
import argparse

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
import pandas as pd

from attack import FGSMAttack, CWAttack
from data.add_gauss_noise import generate_gauss_noise
from model.Cluster import Cluster
from my_data import one_hot_encode, MyDataset, get_error, AdvDataset, get_coordinate, get_asr, FGSMDataset
from model import Model
from torch.utils.tensorboard import SummaryWriter

res1 = []
res2 = []

# runing command
# define all the parameters here
# The running command is : python main.py --generate_model_type ResNet --test_model_type ResNet --adv_test False --adv_train False --pre_cluster False --aux_batch False --epoch_num 15 --learning_rate 0.01 --batch_size 32 --floor 0

parser = argparse.ArgumentParser(description='WiFi Localization')
parser.add_argument('--generate_model_type', type=str, default='ResNet', help='generate model type')
parser.add_argument('--test_model_type', type=str, default='ResNet', help='test model type')
parser.add_argument('--asr_threshold', type=float, default=5, help='ASR threshold')
parser.add_argument('--adv_test', type=str, default='False', help='adv test')
parser.add_argument('--adv_train', type=str, default='False', help='adv train')
parser.add_argument('--aux_batch', type=str, default='False', help='aux batch')
parser.add_argument('--pre_cluster', type=str, default='False', help='pre cluster')
parser.add_argument('--epoch_num', type=int, default=15, help='epoch num')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--floor', type=str, default='0', help='floor')
args = parser.parse_args()
print(args)

generate_model_type = args.generate_model_type
test_model_type = args.test_model_type
asr_threshold = args.asr_threshold

epoch_num = args.epoch_num
learning_rate = args.learning_rate
batch_size = args.batch_size
train_file_path = './data/train' + args.floor + '-103' + '.csv'
test_file_path = './data/test' + args.floor + '-103' + '.csv'
adv_test = True if args.adv_test is 'True' else False
adv_train = True if args.adv_train is 'True' else False
pre_cluster = True if args.pre_cluster is 'True' else False
aux_batch = True if args.aux_batch is 'True' else False
print("adv_test: " + str(adv_test))
print("adv_train: " + str(adv_train))
print("pre_cluster: " + str(pre_cluster))
print("aux_batch: " + str(aux_batch))


# 参数设置
torch.manual_seed(3407)
start_time = datetime.datetime.now()
normal_data_type, adv_data_type, gauss_data_type, adv_data_type2, adv_data_type3, adv_data_type4, adv_data_type5 = \
    'normal', 'adversarial noise', 'gauss noise', 'adversarial noise2', 'adversarial noise3', 'adversarial noise4', 'adversarial noise5'
data_types = [adv_data_type, gauss_data_type, adv_data_type2, adv_data_type3, adv_data_type4, adv_data_type5, normal_data_type]
data_file_path = './data/data.csv'
gauss_file_path = generate_gauss_noise(file_name=train_file_path)
save_path = './weight/weight' + generate_model_type + '_' + floor + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + '.pth'
# 对抗防御是否采用KBN结构
aux_batch = False if not adv_train else aux_batch
# 对抗防御时是否选择多对抗样本L0 + L2 + Linf
multi_data = True
adv_c = 0.01
adv_kappa = 0
# CW生成对抗样本的范数选择, L0, L2, Linf
adv_mode = 'Linf'
adv_increase = False
adv_bias = 0.1
writer = SummaryWriter(
    log_dir='./log/logs' + '-gen-' + generate_model_type + '-test-' + test_model_type + '-epoch-' + str(
        epoch_num) + '-batch-' + str(batch_size) + '-lr-' + str(learning_rate) + '-' + datetime.datetime.now().strftime(
        "%Y%m%d%H%M"))

# 加载数据集
dic, inverse_dic = one_hot_encode(file_path=data_file_path)
dataset = MyDataset(file_path=train_file_path, dic=dic)

cluster = Cluster(file_path=train_file_path, min_clusters=2, max_clusters=2)
cluster_num, new_fila_paths = cluster.kmeans()
print("Number of clusters: " + str(cluster_num))
if pre_cluster:
    for i in range(cluster_num):
        data_types.append(normal_data_type + str(i))
else:
    data_types.append(normal_data_type)
datasets = []
for new_fila_path in new_fila_paths:
    dataset = MyDataset(file_path=new_fila_path, dic=dic)
    datasets.append(dataset)
full_dataset = MyDataset(file_path=train_file_path, dic=dic)
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_batchs = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in datasets]
full_train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_batch = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_batch = DataLoader(
    MyDataset(file_path=test_file_path, dic=dic), batch_size=batch_size, shuffle=False, drop_last=True
)
attack_batch = None
attack_batch1 = None
attack_batch2 = None
attack_batch3 = None
attack_batch4 = None
gauss_batch = None

# 加载模型等
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generate_model = Model(type=generate_model_type, in_channels=1, classes=len(dic), device=device, data_types=data_types).model
test_model = None
if generate_model_type == test_model_type:
    test_model = generate_model
else:
    test_model = Model(type=test_model_type, in_channels=1, classes=len(dic), device=device, data_types=data_types).model
test_model = test_model.to(device)
generate_model = generate_model.to(device)
optimizer = optim.Adam(test_model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()
best_error = 1e9

# 对抗训练预生成对抗样本
if adv_train:
    gauss_batch = DataLoader(
        MyDataset(file_path=gauss_file_path, dic=dic), batch_size=batch_size, shuffle=True, drop_last=True
    )
    print("Generating attack data...")
    attack_dataset = FGSMDataset(
        train_batch=full_train_batch,
        attack=FGSMAttack(epsilon=0.05, device=device),
        model=generate_model,
        device=device, loss_function=loss_function
    )
    attack_batch5 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("Generating attack data...")
    attack_dataset = AdvDataset(
        train_batch=full_train_batch,
        attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='L0',
                        isIncrease=False, bias=0.5)
    )
    attack_batch1 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("Generating attack data...")
    attack_dataset = AdvDataset(
        train_batch=full_train_batch,
        attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='L2',
                        isIncrease=False, bias=0.5)
    )
    attack_batch2 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("Generating attack data...")
    attack_dataset = AdvDataset(
        train_batch=full_train_batch,
        attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='Linf',
                        isIncrease=False, bias=0.5)
    )
    attack_batch3 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("Generating attack data...")
    attack_dataset = AdvDataset(
        train_batch=full_train_batch,
        attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='Linf',
                        isIncrease=True, bias=0.12)
    )
    attack_batch4 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)



gen_data_end_time = datetime.datetime.now()

print("Batch size: {}\nepoch num: {}\n".format(batch_size, epoch_num))
print("Train batch num: {}\nVal batch num:   {}\nTest batch num:  {}".format(
    len(full_train_batch), len(val_batch), len(test_batch))
)

# 训练
for epoch in range(epoch_num):
    # train
    test_model.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    if not adv_train:
        for step, data in enumerate(full_train_batch):
            x, y = data
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = test_model(x)
            loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) \
                            + loss_function(outputs2.view(-1).float(), y[:, 1].float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step + 1) / len(full_train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            writer.add_scalar('Train_Loss_', loss.item(), epoch)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss), end="")
    elif adv_train:
        normal_batch = full_train_batch
        if pre_cluster:
            for step, normal_data in enumerate(zip(*train_batchs)):
                for i in range(len(normal_data)):
                    x, y = normal_data[i]
                    x = x.view(-1, 1, len(x[0])).to(device)
                    y = y.view(-1, len(y[0])).to(device)
                    optimizer.zero_grad()
                    outputs1, outputs2 = test_model(x, normal_data_type + str(i) if aux_batch else normal_data_type)
                    loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) \
                                    + loss_function(outputs2.view(-1).float(), y[:, 1].float())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    rate = (step + 1) / len(full_train_batch)
                    a = "*" * int(rate * 50)
                    b = "." * int((1 - rate) * 50)
                    print("\rNormal \t\ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                        end="")
        for step, (normal_data, adv_data1, adv_data2, adv_data3, adv_data4, adv_data5, gauss_data) in enumerate(
                zip(normal_batch, attack_batch1, attack_batch2, attack_batch3, attack_batch4, attack_batch5, gauss_batch)):
            # if not pre_cluster:
            x, y = normal_data
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = test_model(x, normal_data_type)
            loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) \
                                + loss_function(outputs2.view(-1).float(), y[:, 1].float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step + 1) / len(full_train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rNormal \t\ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                    end="")

            x, y = adv_data1
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = test_model(x, adv_data_type if aux_batch else normal_data_type)
            loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) \
                            + loss_function(outputs2.view(-1).float(), y[:, 1].float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step + 1) / len(full_train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rAdverasial \ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                end="")

            x, y = adv_data2
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = test_model(x, adv_data_type2 if aux_batch else normal_data_type)
            loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) \
                            + loss_function(outputs2.view(-1).float(), y[:, 1].float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step + 1) / len(full_train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rAdverasial \ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                end="")

            x, y = adv_data3
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = test_model(x, adv_data_type3 if aux_batch else normal_data_type)
            loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) \
                            + loss_function(outputs2.view(-1).float(), y[:, 1].float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step + 1) / len(full_train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rAdverasial \ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                end="")

            x, y = adv_data4
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = test_model(x, adv_data_type4 if aux_batch else normal_data_type)
            loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) \
                            + loss_function(outputs2.view(-1).float(), y[:, 1].float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step + 1) / len(full_train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rAdverasial \ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                end="")

            x, y = adv_data5
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = test_model(x, adv_data_type5 if aux_batch else normal_data_type)
            loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) \
                            + loss_function(outputs2.view(-1).float(), y[:, 1].float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step + 1) / len(full_train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rAdverasial \ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                end="")

            x, y = gauss_data
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = test_model(x, gauss_data_type if aux_batch else normal_data_type)
            loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) \
                            + loss_function(outputs2.view(-1).float(), y[:, 1].float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step + 1) / len(full_train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rGauss \t\ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss), end="")

    print("")

    # validate
    test_model.eval()
    err = 0.0
    with torch.no_grad():
        for step, data in enumerate(val_batch):
            x, y = data
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            outputs1, outputs2 = test_model(x, normal_data_type)
            err += torch.sqrt((outputs1.view(-1) - y[:, 0]) ** 2 + (outputs2.view(-1) - y[:, 1]) ** 2).mean()
            # writer.add_scalar('Validation_Loss_' + err, epoch)
        err /= len(val_batch)
        if best_error > err:
            best_error = err
            torch.save(test_model.state_dict(), save_path)
            print('current weight saved!')
        print('[epoch %d] use %f s, validate err: %f m\n' % (epoch + 1, time.perf_counter() - t1, err))

train_end_time = datetime.datetime.now()

rss = []
x = []
y = []

# 测试
err_normal = 0.0
err_adv = 0.0
bias = 0.0
partial_bias = 0.0
test_model.load_state_dict(torch.load(save_path))

t = datetime.datetime.now().strftime("%Y%m%d%H%M")
# writer = csv.writer(csvfile)
# writer.writerow(['RSS' + str(i) for i in range(1, 183)] + ['x'] + ['y'])
res_normal = []

# 测试
test_model.load_state_dict(torch.load(save_path))
for step, data in enumerate(test_batch):
    x, y = data
    x = x.view(-1, 1, len(x[0])).to(device)
    y = y.view(-1, len(y[0]))
    outputs1, outputs2 = test_model(x, normal_data_type)
    outputs1, outputs2 = outputs1.cpu(), outputs2.cpu()
    temp = torch.sqrt((outputs1.view(-1) - y[:, 0]) ** 2 + (outputs2.view(-1) - y[:, 1]) ** 2)
    temp = temp.tolist()
    err_normal += sum(temp) / len(temp)
    res_normal = res_normal + temp

print("Normal prediction's error is {} m".format(err_normal / len(test_batch)))
print("")

if adv_test:
    # Gauss
    err_adv = 0.0
    res_attack = []
    test_model.load_state_dict(torch.load(save_path))
    gauss_file_path = generate_gauss_noise(file_name=test_file_path)
    gauss_batch = DataLoader(
        MyDataset(file_path=gauss_file_path, dic=dic), batch_size=batch_size, shuffle=True, drop_last=True
    )
    for step, data in enumerate(gauss_batch):
        x, y = data
        x = x.view(-1, 1, len(x[0])).to(device)
        y = y.view(-1, len(y[0]))
        outputs1, outputs2 = test_model(x, normal_data_type)
        outputs1, outputs2 = outputs1.cpu(), outputs2.cpu()
        temp = torch.sqrt((outputs1.view(-1) - y[:, 0]) ** 2 + (outputs2.view(-1) - y[:, 1]) ** 2)
        temp = temp.tolist()
        err_adv += sum(temp) / len(temp)
        res_attack = res_attack + temp
    print("Gauss prediction's error is {} m".format(err_adv / len(gauss_batch)))
    print("")

    # FGSM
    accumulated_data_list = []
    columns_rssi = [f"RSSI{i + 1}" for i in range(992)]
    columns_names = columns_rssi + ['x', 'y']
    csv_file_path = './advdata/'
    test_model.load_state_dict(torch.load(save_path))
    fgsmattack = FGSMAttack(epsilon=0.05, device=device)
    err_adv = 0.0
    bias = 0.0
    partial_bias = 0.0
    res_attack = []
    for step, data in enumerate(test_batch):
        x, y = data
        x = x.view(-1, 1, len(x[0])).to(device)
        y = y.view(-1, len(y[0])).to(device)
        x.requires_grad = True
        outputs1, outputs2 = generate_model(x, normal_data_type)
        loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) + loss_function(outputs2.view(-1).float(), y[:, 1].float())
        generate_model.zero_grad()
        loss.backward()
        grad_sign = x.grad.data.sign().view(len(x), -1).cpu()
        x = x.view(len(x), -1).cpu()
        x_adv = fgsmattack.generate_data(x, grad_sign)
        x_adv = x_adv.float()
        x_adv = x_adv.view(-1, 1, len(x_adv[0])).to(device)
        y_adv1, y_adv2 = test_model(x_adv, normal_data_type)
        bias += torch.abs(torch.abs(x_adv.cpu() - x) / x).mean()
        abs_diff = torch.abs(x_adv.cpu() - x)
        partial_bias += torch.abs((abs_diff / x))[abs_diff != 0].mean()
        temp = torch.sqrt((y_adv1.view(-1) - y[:, 0]) ** 2 + (y_adv2.view(-1) - y[:, 1]) ** 2)
        temp = temp.tolist()
        err_adv += sum(temp) / len(temp)
        res_attack = res_attack + temp
        combined_data_row = np.concatenate((x_adv.cpu().view(len(x_adv), -1), data[1]), axis=1)
        accumulated_data_list.append(combined_data_row)

    accumulated_data = np.vstack(accumulated_data_list)
    df = pd.DataFrame(accumulated_data, columns=columns_names)
    csv_file_name = 'test_FGSM_' + generate_model_type + floor + '.csv'
    df.to_csv(csv_file_path + csv_file_name, index=False)
    print('{}-attack file is saved in {}'.format('FGSM', csv_file_path + csv_file_name))
    print("After {} attack, prediction's Error is {} m".format('FGSM', err_adv / len(test_batch)))
    print("After {} attack, ASR(threshold = {}) is {} %".format('FGSM', asr_threshold,
                                                                get_asr(res_normal, res_attack, asr_threshold) * 100))
    # print("Attack bias: {}%".format(bias / len(test_batch) * 100))
    # print("Partial attack bias: {}%".format(partial_bias / len(test_batch) * 100))
    print("")
    # CW
    accumulated_data_list = []
    test_model.load_state_dict(torch.load(save_path))
    err_adv = 0.0
    bias = 0.0
    partial_bias = 0.0
    res_attack = []
    attack = CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='Linf', isIncrease=True,
                    bias=0.1)
    for step, data in enumerate(test_batch):
        x, y = data
        x = x.view(-1, 1, len(x[0])).to(device)
        y = y.view(-1, len(y[0]))

        x, y = data
        x_adv, y_adv = attack.generate_data(data)
        x_adv = x_adv.view(-1, 1, len(x[0])).to(device)
        y_adv = y_adv.view(-1, len(y[0])).to(device)
        bias += torch.abs(torch.abs(x_adv.cpu() - x) / x).mean()
        abs_diff = torch.abs(x_adv.cpu() - x)
        partial_bias += torch.abs((abs_diff / x))[abs_diff != 0].mean()
        y_adv1, y_adv2 = test_model(x_adv, normal_data_type)
        temp = torch.sqrt((y_adv1.view(-1) - y_adv[:, 0]) ** 2 + (y_adv2.view(-1) - y_adv[:, 1]) ** 2)
        temp = temp.tolist()
        err_adv += sum(temp) / len(temp)
        res_attack = res_attack + temp
        combined_data_row = np.concatenate((x_adv.cpu().view(len(x_adv), -1), data[1]), axis=1)
        accumulated_data_list.append(combined_data_row)

    accumulated_data = np.vstack(accumulated_data_list)
    df = pd.DataFrame(accumulated_data, columns=columns_names)
    csv_file_name = 'test_CW_' + generate_model_type + floor + '.csv'
    df.to_csv(csv_file_path + csv_file_name, index=False)
    print('{}-attack file is saved in {}'.format('CW', csv_file_path + csv_file_name))
    print("After {} attack, prediction's Error is {} m".format('CW', err_adv / len(test_batch)))
    print("After {} attack, ASR(threshold = {}) is {} %".format('CW', asr_threshold,
                                                                get_asr(res_normal, res_attack,
                                                                        asr_threshold) * 100))
    print("Attack bias: {}%".format(bias / len(test_batch) * 100))
    print("Partial attack bias: {}%".format(partial_bias / len(test_batch) * 100))
    print("")

    # Ours
    for adv_mode, adv_bias in zip(['L0', 'L2', 'Linf'], [0.5, 0.5, 0.5]):
        accumulated_data_list = []
        test_model.load_state_dict(torch.load(save_path))
        err_adv = 0.0
        bias = 0.0
        partial_bias = 0.0
        res_attack = []
        attack = CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode=adv_mode, isIncrease=False,
                        bias=adv_bias)
        for step, data in enumerate(test_batch):
            x, y = data
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0]))

            x, y = data
            x_adv, y_adv = attack.generate_data(data)
            x_adv = x_adv.view(-1, 1, len(x[0])).to(device)
            y_adv = y_adv.view(-1, len(y[0])).to(device)
            bias += torch.abs(torch.abs(x_adv.cpu() - x) / x).mean()
            abs_diff = torch.abs(x_adv.cpu() - x)
            partial_bias += torch.abs((abs_diff / x))[abs_diff != 0].mean()
            y_adv1, y_adv2 = test_model(x_adv, normal_data_type)
            temp = torch.sqrt((y_adv1.view(-1) - y_adv[:, 0]) ** 2 + (y_adv2.view(-1) - y_adv[:, 1]) ** 2)
            temp = temp.tolist()
            err_adv += sum(temp) / len(temp)
            res_attack = res_attack + temp
            combined_data_row = np.concatenate((x_adv.cpu().view(len(x_adv), -1), data[1]), axis=1)
            accumulated_data_list.append(combined_data_row)

        accumulated_data = np.vstack(accumulated_data_list)
        df = pd.DataFrame(accumulated_data, columns=columns_names)
        csv_file_name = 'test_' + adv_mode + '_' + generate_model_type + floor + '.csv'
        df.to_csv(csv_file_path + csv_file_name, index=False)
        print('{}-attack file is saved in {}'.format(adv_mode, csv_file_path + csv_file_name))
        print("After {} attack, prediction's Error is {} m".format(adv_mode, err_adv / len(test_batch)))
        print("After {} attack, ASR(threshold = {}) is {} %".format(adv_mode, asr_threshold,
                                                                    get_asr(res_normal, res_attack,
                                                                            asr_threshold) * 100))
        print("Attack bias: {}%".format(bias / len(test_batch) * 100))
        print("Partial attack bias: {}%".format(partial_bias / len(test_batch) * 100))
        print("")

test_end_time = datetime.datetime.now()

res1.append(err_normal / len(test_batch))

with open('./log/log-' + '-gen-' + generate_model_type + '-test-' + test_model_type + '-epoch-' + str(
        epoch_num) + '-lr-' + str(learning_rate) + '-' + t + '.txt', 'a') as f:
    f.write("########################\n")
    f.write("Parameters:\n")
    f.write(f"Gerenate Model: {generate_model_type}\n")
    f.write(f"Test Model: {test_model_type}\n")
    f.write(f"\tbatch size: {batch_size}\n")
    f.write(f"\tepoch num: {epoch_num}\n")
    if adv_train:
        f.write(f"\tadv test: {adv_test}\n")
    f.write(f"\tadv train: {adv_train}\n")
    f.write(f"\taux batch: {aux_batch}\n")
    f.write(f"\tnmulti data: {multi_data}\n")
    f.write(f"\tadv mode: {adv_mode}\n")
    f.write(f"\tadv increase: {adv_increase}\n")
    f.write(f"\tadv bias: {adv_bias}\n")
    f.write(f"\ttrain batch size(only count normal data): {len(full_train_batch)},"
            f"\tval batch size: {len(val_batch)},"
            f"\ttest batch size: {len(test_batch)}\n")
    f.write("------------------------\n")
    f.write("Execution time information\n")
    f.write(f"\tStart time: {start_time}\n")
    f.write(f"\tGenerating data uses {gen_data_end_time - start_time}\n")
    f.write(f"\tTrain uses {train_end_time - gen_data_end_time}\n")
    f.write(f"\tTest uses {test_end_time - train_end_time}\n")
    f.write(f"\tEnd time: {test_end_time}\n")
    f.write("------------------------\n")
    f.write("Result:\n")
    f.write(f"\tnormal prediction error: {err_normal / len(test_batch)} m\n")
    if adv_test:
        f.write(f"\tattack prediction error: {err_adv / len(test_batch)} m\n")
        f.write(f"\tAttack bias: {bias / len(test_batch) * 100}% \n")
        f.write(f"\tPartial attack bias: {partial_bias / len(test_batch) * 100}% \n")
    f.write("########################\n\n")
