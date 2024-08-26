import csv
import datetime
import time

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split

from attack import CWAttack
from data.add_gauss_noise import generate_gauss_noise
from my_data import one_hot_encode, MyDataset, get_error, AdvDataset, get_coordinate
from model import Model

res1 = []
res2 = []

torch.manual_seed(3407)
start_time = datetime.datetime.now()
normal_data_type, adv_data_type, gauss_data_type, adv_data_type2, adv_data_type3 = \
    'normal', 'adversarial noise', 'gauss noise', 'adversarial noise2', 'adversarial noise3'
train_file_path = '../data/train.csv'
gauss_file_path = '../data/gauss_data.csv'
test_file_path = '../data/test.csv'
save_path = './weight_my.pth'
batch_size = 32
epoch_num = 50
adv_test = True
adv_train = False
aux_batch = False if not adv_train else True
multi_data = True
adv_c = 0.1
adv_kappa = 0
adv_mode = 'Linf'
adv_increase = True
adv_bias = 0.1
generate_model_type = 'ResNet'
test_model_type = 'ResNet'

dic, inverse_dic = one_hot_encode(file_path=train_file_path)

dataset = MyDataset(file_path=train_file_path, dic=dic)

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
generate_model = Model(type=generate_model_type, in_channels=1, classes=len(dic), device=device).model
test_model = None
if generate_model_type == test_model_type:
    test_model = generate_model
else:
    test_model = Model(type=test_model_type, in_channels=1, classes=len(dic), device=device).model
test_model = test_model.to(device)
generate_model = generate_model.to(device)
optimizer = optim.Adam(test_model.parameters(), lr=0.0002)
loss_function = nn.CrossEntropyLoss()
best_error = 1e9

if adv_train:
    generate_gauss_noise(train_file_path)
    gauss_batch = DataLoader(
        MyDataset(file_path=gauss_file_path, dic=dic), batch_size=batch_size, shuffle=True, drop_last=True
    )
    if not multi_data:
        print("Generating attack data...")
        attack_dataset = AdvDataset(
            train_batch=train_batch,
            attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode=adv_mode,
                            isIncrease=adv_increase, bias=adv_bias)
        )
        attack_batch = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        print("Generating attack data...")
        attack_dataset = AdvDataset(
            train_batch=train_batch,
            attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='L0',
                            isIncrease=True, bias=0.1)
        )
        attack_batch1 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        print("Generating attack data...")
        attack_dataset = AdvDataset(
            train_batch=train_batch,
            attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='L2',
                            isIncrease=True, bias=0.3)
        )
        attack_batch2 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        print("Generating attack data...")
        attack_dataset = AdvDataset(
            train_batch=train_batch,
            attack=CWAttack(generate_model, device, c=adv_c, kappa=adv_kappa, mode='Linf',
                            isIncrease=True, bias=0.1)
        )
        attack_batch3 = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

gen_data_end_time = datetime.datetime.now()

print("Batch size: {}\nepoch num: {}\n".format(batch_size, epoch_num))
print("Train batch num: {}\nVal batch num:   {}\nTest batch num:  {}".format(
    len(train_batch), len(val_batch), len(test_batch))
)

for epoch in range(epoch_num):
    # train
    test_model.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    if not adv_train:
        for step, data in enumerate(train_batch):
            x, y = data
            x = x.view(-1, 1, len(x[0])).to(device)
            y = y.view(-1, len(y[0])).to(device)
            optimizer.zero_grad()
            outputs = test_model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step + 1) / len(train_batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss), end="")
    elif adv_train:
        if not multi_data:
            for step, (normal_data, adv_data) in enumerate(zip(train_batch, attack_batch)):
                x, y = normal_data
                x = x.view(-1, 1, len(x[0])).to(device)
                y = y.view(-1, len(y[0])).to(device)
                optimizer.zero_grad()
                outputs = test_model(x, normal_data_type)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                rate = (step + 1) / len(train_batch)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rNormal \t\ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                      end="")

                x, y = adv_data
                x = x.view(-1, 1, len(x[0])).to(device)
                y = y.view(-1, len(y[0])).to(device)
                optimizer.zero_grad()
                outputs = test_model(x, adv_data_type if aux_batch else normal_data_type)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                rate = (step + 1) / len(train_batch)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rAdverasial \ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                      end="")
        else:
            for step, (normal_data, adv_data1, adv_data2, adv_data3, gauss_data) in enumerate(
                    zip(train_batch, attack_batch1, attack_batch2, attack_batch3, gauss_batch)):
                x, y = normal_data
                x = x.view(-1, 1, len(x[0])).to(device)
                y = y.view(-1, len(y[0])).to(device)
                optimizer.zero_grad()
                outputs = test_model(x, normal_data_type)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                rate = (step + 1) / len(train_batch)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rNormal \t\ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                      end="")

                x, y = adv_data1
                x = x.view(-1, 1, len(x[0])).to(device)
                y = y.view(-1, len(y[0])).to(device)
                optimizer.zero_grad()
                outputs = test_model(x, adv_data_type if aux_batch else normal_data_type)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                rate = (step + 1) / len(train_batch)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rAdverasial \ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                      end="")

                x, y = adv_data2
                x = x.view(-1, 1, len(x[0])).to(device)
                y = y.view(-1, len(y[0])).to(device)
                optimizer.zero_grad()
                outputs = test_model(x, adv_data_type2 if aux_batch else normal_data_type)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                rate = (step + 1) / len(train_batch)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rAdverasial \ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                      end="")

                x, y = adv_data3
                x = x.view(-1, 1, len(x[0])).to(device)
                y = y.view(-1, len(y[0])).to(device)
                optimizer.zero_grad()
                outputs = test_model(x, adv_data_type3 if aux_batch else normal_data_type)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                rate = (step + 1) / len(train_batch)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rAdverasial \ttrain loss: {:^3.0f}%[{}->{}]{:.6f}".format(int(rate * 100), a, b, loss),
                      end="")

                x, y = gauss_data
                x = x.view(-1, 1, len(x[0])).to(device)
                y = y.view(-1, len(y[0])).to(device)
                optimizer.zero_grad()
                outputs = test_model(x, gauss_data_type if aux_batch else normal_data_type)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                rate = (step + 1) / len(train_batch)
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
            outputs = test_model(x, normal_data_type)
            err += get_error(outputs.cpu(), y.cpu(), inverse_dic)
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

# test
err_normal = 0.0
err_adv = 0.0
bias = 0.0
partial_bias = 0.0
attack = CWAttack(test_model, device, c=adv_c, kappa=adv_kappa, mode=adv_mode, isIncrease=adv_increase, bias=adv_bias)
test_model.load_state_dict(torch.load(save_path))

t = datetime.datetime.now().strftime("%Y%m%d%H%M")
with open('../data/attack-' + t + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['RSS' + str(i) for i in range(1, 183)] + ['x'] + ['y'])
    for step, data in enumerate(test_batch):
        x, y = data
        x = x.view(-1, 1, len(x[0])).to(device)
        y = y.view(-1, len(y[0]))

        outputs = test_model(x, normal_data_type).cpu()
        err_normal += get_error(outputs, y, inverse_dic)

        x, y = data

        # write test data and result
        if adv_test:
            x_adv, y_adv = attack.generate_data(data)
            x_adv1 = x_adv.view(-1, 1, len(x[0])).to(device)
            y_adv1 = y_adv.view(-1, len(y[0]))
            bias += torch.abs(torch.abs(x_adv.cpu() - x) / x).mean()
            abs_diff = torch.abs(x_adv.cpu() - x)
            partial_bias += torch.abs((abs_diff / x))[abs_diff != 0].mean()
            outputs_adv = test_model(x_adv1, normal_data_type).cpu()
            err_adv += get_error(outputs_adv, y_adv1, inverse_dic)

            # write data
            outputs = outputs.detach()
            outputs_adv = outputs_adv.detach()
            for i in range(len(x)):
                temp = inverse_dic[int(np.argmax(y[i])) + 1]
                x_abs, y_abs = temp[0], temp[1]
                writer.writerow(x.tolist()[i] + [x_abs] + [y_abs])

                x_predict, y_predict = get_coordinate(outputs[i], inverse_dic)
                writer.writerow(x.tolist()[i] + [x_predict] + [y_predict])

                x_predict, y_predict = get_coordinate(outputs_adv[i], inverse_dic)
                writer.writerow(x_adv.tolist()[i] + [x_predict] + [y_predict])

test_end_time = datetime.datetime.now()

print("Normal prediction's error is {} m".format(err_normal / len(test_batch)))
res1.append(err_normal / len(test_batch))
if adv_test:
    print("After attack, prediction's Error is {} m".format(err_adv / len(test_batch)))
    print("Attack bias: {}%".format(bias / len(test_batch) * 100))
    print("Partial attack bias: {}%".format(partial_bias / len(test_batch) * 100))
    res2.append(err_adv / len(test_batch))

with open('../log/log-' + t + '.txt', 'a') as f:
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
    f.write(f"\ttrain batch size(only count normal data): {len(train_batch)},"
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

print(sum(res1) / len(res1))
print(sum(res2) / len(res2))
