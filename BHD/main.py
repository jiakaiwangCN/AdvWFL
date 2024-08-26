import csv
import datetime
import time

import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

from attack import CWAttack, FGSMAttack
from data.add_gauss_noise import generate_gauss_noise
from model.Cluster import Cluster
from my_data import one_hot_encode, MyDataset, get_error, AdvDataset, get_coordinate, get_asr
from model import Model


torch.manual_seed(3407)
start_time = datetime.datetime.now()
normal_data_type, adv_data_type, gauss_data_type, adv_data_type2, adv_data_type3 = \
    'normal', 'adversarial noise', 'gauss noise', 'adversarial noise2', 'adversarial noise3'
data_types = [adv_data_type, gauss_data_type, adv_data_type2, adv_data_type3, normal_data_type]
train_file_path = './data/train.csv'
gauss_file_path = './data/gauss_data.csv'
test_file_path = './data/test.csv'
save_path = './weight_my.pth'
batch_size = 32
epoch_num = 50
adv_test = True
adv_train = False
aux_batch = False if not adv_train else True
pre_cluster = False if not adv_train else True
multi_data = True
adv_c = 0
adv_kappa = 0
generate_model_type = 'ResNet'
test_model_type = 'ResNet'
asr_threshold = 2.5
dic, inverse_dic = one_hot_encode(file_path=train_file_path)

cluster = Cluster(file_path=train_file_path, min_clusters=2, max_clusters=5)
cluster_num, new_fila_paths = cluster.kmeans()
print("Number of clusters: " + str(cluster_num))

if pre_cluster:
    for i in range(cluster_num):
        data_types.append(normal_data_type + str(i))

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
gauss_batch = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generate_model = Model(type=generate_model_type, in_channels=1, classes=len(dic), device=device,
                        data_types=data_types).model
test_model = None
print(data_types)
if generate_model_type == test_model_type:
    test_model = generate_model
else:
    test_model = Model(type=test_model_type, in_channels=1, classes=len(dic), device=device,
                        data_types=data_types).model
test_model = test_model.to(device)
generate_model = generate_model.to(device)
optimizer = optim.Adam(test_model.parameters(), lr=0.0005)
scaler = amp.GradScaler()
loss_function = nn.MSELoss()
best_error = 1e9

if adv_train:
    generate_gauss_noise(train_file_path)
    gauss_batch = DataLoader(
        MyDataset(file_path=gauss_file_path, dic=dic), batch_size=batch_size, shuffle=True, drop_last=True
    )
    if not multi_data:
        print("Generating attack data...")
        attack_dataset = AdvDataset(
            train_batch=full_train_batch,
            attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode=adv_mode,
                            isIncrease=False, bias=adv_bias)
        )
        attack_batch = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        print("Generating attack data...")
        attack_dataset = AdvDataset(
            train_batch=full_train_batch,
            attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='L0',
                            isIncrease=False, bias=0.2)
        )
        attack_batch1 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        print("Generating attack data...")
        attack_dataset = AdvDataset(
            train_batch=full_train_batch,
            attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='L2',
                            isIncrease=False, bias=0.2)
        )
        attack_batch2 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        print("Generating attack data...")
        attack_dataset = AdvDataset(
            train_batch=full_train_batch,
            attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='Linf',
                            isIncrease=False, bias=0.2)
        )
        attack_batch3 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

gen_data_end_time = datetime.datetime.now()

print("Batch size: {}\nepoch num: {}\n".format(batch_size, epoch_num))
print("Train batch num: {}\nVal batch num:   {}\nTest batch num:  {}".format(
    len(full_train_batch), len(val_batch), len(test_batch))
)

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
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward() 
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            rate = (step + 1) / len(full_train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
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
                    print(
                        "\rNormal \t\ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                        end="")
        for step, (normal_data, adv_data1, adv_data2, adv_data3, gauss_data) in enumerate(
                zip(normal_batch, attack_batch1, attack_batch2, attack_batch3, gauss_batch)):
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
            outputs1, outputs2 = test_model(x)
            outputs1, outputs2 = test_model(x, normal_data_type)
            err += torch.sqrt((outputs1.view(-1) - y[:, 0]) ** 2 + (outputs2.view(-1) - y[:, 1]) ** 2).mean()
        err /= len(val_batch)
        if best_error > err:
            best_error = err
            torch.save(test_model.state_dict(), save_path)
            print('current weight saved!')
        print('[epoch %d] use %f s, validate err: %f m\n' % (epoch + 1, time.perf_counter() - t1, err))

train_end_time = datetime.datetime.now()

res_normal = []

# 测试
test_model.load_state_dict(torch.load(save_path))
err_normal = 0.0
pool_values = []
for step, data in enumerate(test_batch):
    x, y = data
    x = x.view(-1, 1, len(x[0])).to(device)
    y = y.view(-1, len(y[0]))
    outputs1, outputs2, pool_value = test_model(x, normal_data_type, ret_pool_value=True)
    outputs1, outputs2 = outputs1.cpu(), outputs2.cpu()
    temp = torch.sqrt((outputs1.view(-1) - y[:, 0]) ** 2 + (outputs2.view(-1) - y[:, 1]) ** 2)
    temp = temp.tolist()
    err_normal += sum(temp) / len(temp)
    res_normal = res_normal + temp
    pool_values.append(pool_value)

pool_value = torch.cat(pool_values, dim=0).cpu().detach().numpy()
np.save('tsne_pool_value_normal.npy', pool_value)
print("Normal prediction's error is {} m".format(err_normal / len(test_batch)))
print("")

test_model.load_state_dict(torch.load(save_path))
if adv_test:
    normal_datas = []
    attack_datas = []
    pool_values = []
    accumulated_data_list = []
    columns_rssi = [f"RSSI{i + 1}" for i in range(182)]
    columns_names = columns_rssi + ['x', 'y']
    csv_file_path = './classic_data/'
    # Gauss
    gauss_batch = DataLoader(
        MyDataset(file_path='./data/gauss_data_test.csv', dic=dic), batch_size=batch_size, shuffle=True,
        drop_last=True
    )
    err_adv = 0.0
    bias = 0.0
    partial_bias = 0.0
    res_attack = []
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
    print("After {} attack, prediction's Error is {} m".format('Gauss', err_adv / len(test_batch)))
    print("After {} attack, ASR(threshold = {}) is {} %".format('Gauss', asr_threshold,
                                                                get_asr(res_normal, res_attack,
                                                                        asr_threshold) * 100))
    print("")
    # FGSM
    pool_values = []
    accumulated_data_list = []
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
        loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) + loss_function(outputs2.view(-1).float(),
                                                                                            y[:, 1].float())
        generate_model.zero_grad()
        loss.backward()
        grad_sign = x.grad.data.sign().view(len(x), -1).cpu()
        x = x.view(len(x), -1).cpu()
        x_adv = fgsmattack.generate_data(x, grad_sign)
        x_adv = x_adv.float()
        x_adv = x_adv.view(-1, 1, len(x_adv[0])).to(device)
        y_adv1, y_adv2, pool_value = test_model(x_adv, normal_data_type, ret_pool_value=True)
        bias += torch.abs(torch.abs(x_adv.cpu() - x) / x).mean()
        abs_diff = torch.abs(x_adv.cpu() - x)
        partial_bias += torch.abs((abs_diff / x))[abs_diff != 0].mean()
        temp = torch.sqrt((y_adv1.view(-1) - y[:, 0]) ** 2 + (y_adv2.view(-1) - y[:, 1]) ** 2)
        temp = temp.tolist()
        err_adv += sum(temp) / len(temp)
        res_attack = res_attack + temp
        combined_data_row = np.concatenate((x_adv.cpu().view(len(x_adv), -1), data[1]), axis=1)
        accumulated_data_list.append(combined_data_row)
        pool_values.append(pool_value)

    accumulated_data = np.vstack(accumulated_data_list)
    df = pd.DataFrame(accumulated_data, columns=columns_names)
    csv_file_name = 'test_FGSM_' + generate_model_type + '.csv'
    df.to_csv(csv_file_path + csv_file_name, index=False)
    print('{}-attack file is saved in {}'.format('FGSM', csv_file_path + csv_file_name))
    print("After {} attack, prediction's Error is {} m".format('FGSM', err_adv / len(test_batch)))
    print("After {} attack, ASR(threshold = {}) is {} %".format('FGSM', asr_threshold,
                                                                get_asr(res_normal, res_attack,
                                                                        asr_threshold) * 100))
    print("Attack bias: {}%".format(bias / len(test_batch) * 100))
    print("Partial attack bias: {}%".format(partial_bias / len(test_batch) * 100))
    print("")
    normal_datas.append(res_normal)
    attack_datas.append(res_attack)
    pool_value = torch.cat(pool_values, dim=0).cpu().detach().numpy()
    np.save('tsne_pool_value_FGSM.npy', pool_value)
    # CW
    pool_values = []
    accumulated_data_list = []
    err_adv = 0.0
    bias = 0.0
    partial_bias = 0.0
    res_attack = []
    attack = CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='Linf', isIncrease=True,
                        bias=0.15)
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
        y_adv1, y_adv2, pool_value = test_model(x_adv, normal_data_type, ret_pool_value=True)
        temp = torch.sqrt((y_adv1.view(-1) - y_adv[:, 0]) ** 2 + (y_adv2.view(-1) - y_adv[:, 1]) ** 2)
        temp = temp.tolist()
        err_adv += sum(temp) / len(temp)
        res_attack = res_attack + temp
        combined_data_row = np.concatenate((x_adv.cpu().view(len(x_adv), -1), data[1]), axis=1)
        accumulated_data_list.append(combined_data_row)
        pool_values.append(pool_value)

    accumulated_data = np.vstack(accumulated_data_list)
    df = pd.DataFrame(accumulated_data, columns=columns_names)
    csv_file_name = 'test_CW_' + generate_model_type + '.csv'
    df.to_csv(csv_file_path + csv_file_name, index=False)
    print('{}-attack file is saved in {}'.format('CW', csv_file_path + csv_file_name))
    print("After {} attack, prediction's Error is {} m".format('CW', err_adv / len(test_batch)))
    print("After {} attack, ASR(threshold = {}) is {} %".format('CW', asr_threshold,
                                                                get_asr(res_normal, res_attack,
                                                                        asr_threshold) * 100))
    print("Attack bias: {}%".format(bias / len(test_batch) * 100))
    print("Partial attack bias: {}%".format(partial_bias / len(test_batch) * 100))
    print("")
    normal_datas.append(res_normal)
    attack_datas.append(res_attack)
    pool_value = torch.cat(pool_values, dim=0).cpu().detach().numpy()
    np.save('tsne_pool_value_CW.npy', pool_value)
    # Ours
    for adv_mode, adv_bias in zip(['L0', 'L2', 'Linf'], [0.5, 0.5, 0.5]):
        pool_values = []
        accumulated_data_list = []
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
            y_adv1, y_adv2, pool_value = test_model(x_adv, normal_data_type, ret_pool_value=True)
            temp = torch.sqrt((y_adv1.view(-1) - y_adv[:, 0]) ** 2 + (y_adv2.view(-1) - y_adv[:, 1]) ** 2)
            temp = temp.tolist()
            err_adv += sum(temp) / len(temp)
            res_attack = res_attack + temp
            combined_data_row = np.concatenate((x_adv.cpu().view(len(x_adv), -1), data[1]), axis=1)
            accumulated_data_list.append(combined_data_row)
            pool_values.append(pool_value)

        accumulated_data = np.vstack(accumulated_data_list)
        df = pd.DataFrame(accumulated_data, columns=columns_names)
        csv_file_name = 'test_' + adv_mode + '_' + generate_model_type + '.csv'
        df.to_csv(csv_file_path + csv_file_name, index=False)
        print('{}-attack file is saved in {}'.format(adv_mode, csv_file_path + csv_file_name))
        print("After {} attack, prediction's Error is {} m".format(adv_mode, err_adv / len(test_batch)))
        print("After {} attack, ASR(threshold = {}) is {} %".format(adv_mode, asr_threshold,
                                                                    get_asr(res_normal, res_attack,
                                                                            asr_threshold) * 100))
        print("Attack bias: {}%".format(bias / len(test_batch) * 100))
        print("Partial attack bias: {}%".format(partial_bias / len(test_batch) * 100))
        print("")
        normal_datas.append(res_normal)
        attack_datas.append(res_attack)
        pool_value = torch.cat(pool_values, dim=0).cpu().detach().numpy()
        np.save('tsne_pool_value_' + adv_mode + '.npy', pool_value)
t = datetime.datetime.now().strftime("%Y%m%d%H%M")
# with open('./data/attack-' + t + '.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['RSS' + str(i) for i in range(n=3, 183)] + ['x'] + ['y'])
#     for step, data in enumerate(test_batch):
#         x, y = data
#         x = x.view(-n=3, n=3, len(x[0])).to(device)
#         y = y.view(-n=3, len(y[0]))
#
#         outputs = test_model(x, normal_data_type).cpu()
#         err_normal += get_error(outputs, y, inverse_dic)
#
#         x, y = data
#
#         # write test data and result
#         if adv_test:
#             x_adv, y_adv = attack.generate_data(data)
#             x_adv1 = x_adv.view(-n=3, n=3, len(x[0])).to(device)
#             y_adv1 = y_adv.view(-n=3, len(y[0]))
#             bias += torch.abs(torch.abs(x_adv.cpu() - x) / x).mean()
#             abs_diff = torch.abs(x_adv.cpu() - x)
#             partial_bias += torch.abs((abs_diff / x))[abs_diff != 0].mean()
#             outputs_adv = test_model(x_adv1, normal_data_type).cpu()
#             err_adv += get_error(outputs_adv, y_adv1, inverse_dic)
#
#             # write data
#             outputs = outputs.detach()
#             outputs_adv = outputs_adv.detach()
#             for i in range(len(x)):
#                 temp = inverse_dic[int(np.argmax(y[i])) + n=3]
#                 x_abs, y_abs = temp[0], temp[n=3]
#                 writer.writerow(x.tolist()[i] + [x_abs] + [y_abs])
#
#                 x_predict, y_predict = get_coordinate(outputs[i], inverse_dic)
#                 writer.writerow(x.tolist()[i] + [x_predict] + [y_predict])
#
#                 x_predict, y_predict = get_coordinate(outputs_adv[i], inverse_dic)
#                 writer.writerow(x_adv.tolist()[i] + [x_predict] + [y_predict])

test_end_time = datetime.datetime.now()

with open('./log/log-' + t + '.txt', 'a') as f:
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
    f.write(f"\tmulti data: {multi_data}\n")
    f.write(f"\tpre cluster: {pre_cluster}\n")
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
