import math
import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import optim, nn

from physical.model import get_physical_model


class Attack:
    def __init__(self, model, device, c=0.1, kappa=2, max_iter=300, learning_rate=0.01, mode='L2', isIncrease=False,
                 bias=0.05):
        self.mode = mode
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.kappa = kappa
        self.device = device
        self.model = model
        self.c = c
        self.isIncrease = isIncrease
        self.bias = bias
        os.chdir('../physical')
        self.physical_model = get_physical_model()
        os.chdir('../demo')

    def generate_data(self, data):
        """

        :param data: DataLoader
        :return: DataLoader being attacked
        """

        y = torch.argmax(data[1], dim=1)
        x_attack, best_layers1, best_layers2 = self.cw_l2_attack(
            model=self.model, images=data[0], labels=y, site=data[2], device=self.device, c=self.c, kappa=self.kappa,
            max_iter=self.max_iter, learning_rate=self.learning_rate, mode=self.mode, isIncrease=self.isIncrease,
            bias=self.bias
        )
        # same = x_attack == data[0]
        # print("\n{} elements changed".fomat(np.count_nonzero(same == False)))
        return x_attack.float(), data[1], best_layers1, best_layers2

    def cw_l2_attack(self, model, images, site, labels, device, targeted=False, c=0.1, kappa=2, max_iter=100,
                     learning_rate=0.01, mode='L2', isIncrease=True, bias=0.05):

        class CustomMSELoss(nn.Module):
            def __init__(self):
                super(CustomMSELoss, self).__init__()

            def forward(self, a, images, mask):
                loss = ((mask * (a - images)) ** 2).sum()
                return loss

        # Define f-function
        def f(x):
            outputs = model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())

            if targeted:
                return torch.clamp(i - j, min=-kappa)
            else:
                return torch.clamp(j - i, min=-kappa)

        assert mode in ['L0', 'L2', 'Linf']

        images_lower_bound = (images - torch.abs(images) * bias).clamp(torch.min(images), torch.max(images))
        images_upper_bound = (images + torch.abs(images) * bias).clamp(torch.min(images), torch.max(images))

        scaler = MinMaxScaler(feature_range=(0, 1))
        init_images = images
        images = torch.from_numpy(scaler.fit_transform(images.transpose(0, 1))).transpose(0, 1)
        images = images.view(-1, 1, len(images[0])).float()
        images = images.to(device)
        images_lower_bound = torch.from_numpy(scaler.transform(images_lower_bound.transpose(0, 1))).transpose(0, 1)
        images_lower_bound = images_lower_bound.view(-1, 1, len(images_lower_bound[0])).float()
        images_lower_bound = images_lower_bound.to(device)
        images_upper_bound = torch.from_numpy(scaler.transform(images_upper_bound.transpose(0, 1))).transpose(0, 1)
        images_upper_bound = images_upper_bound.view(-1, 1, len(images_upper_bound[0])).float()
        images_upper_bound = images_upper_bound.to(device)
        labels = labels.to(device)

        w = torch.torch.full_like(images, -1).to(device)
        w.requires_grad = True
        w_lower_bound = 1 / 2 * torch.log(-(images_lower_bound / (images_lower_bound - 1)))
        w_lower_bound = w_lower_bound.to(device)
        w_upper_bound = 1 / 2 * torch.log(-(images_upper_bound / (images_upper_bound - 1)))
        w_upper_bound = torch.where(torch.isnan(w_upper_bound), torch.inf, w_upper_bound)
        w_upper_bound = w_upper_bound.to(device)
        optimizer = optim.Adam([w], lr=learning_rate)
        best_images = None
        best_layer1 = [0 for _ in range(len(init_images))]
        best_layer2 = [0 for _ in range(len(init_images))]
        for step in range(max_iter):
            a = 1 / 2 * (nn.Tanh()(w) + 1)
            a = a.to(device)
            # 第一个目标，对抗样本与原始样本足够接近
            loss1 = None
            mask = a > images
            mask = mask.float() + 0.01 * (~mask).float()
            bias_mask = torch.logical_and(a < images_upper_bound, a > images_lower_bound)
            bias_mask_not = torch.logical_not(bias_mask)

            if mode == 'L2':
                if isIncrease:
                    loss1 = None
                    if step > max_iter * 0.5:
                        loss1 = 0.01 * torch.nn.MSELoss(reduction='sum')(bias_mask * a, bias_mask * images)
                        loss1 = loss1 + torch.nn.MSELoss(reduction='sum')(bias_mask_not * a, bias_mask_not * images)
                    else:
                        loss1 = torch.nn.MSELoss(reduction='sum')(a, images)
                else:
                    if step > max_iter * 0.5:
                        loss1 = 0.01 * CustomMSELoss()(bias_mask * a, bias_mask * images, mask)
                        loss1 = loss1 + CustomMSELoss()(bias_mask_not * a, bias_mask_not * images, mask)
                    else:
                        loss1 = torch.nn.MSELoss(reduction='sum')(a, images)

            # 第二个目标，误导模型输出
            loss2 = torch.sum(c * f(a))
            cost = loss1 + loss2
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if step < max_iter / 2:
                continue
            attack_image = torch.from_numpy(
                scaler.inverse_transform(a.cpu().detach().view(a.shape[0], -1).transpose(0, 1))).transpose(0, 1)
            physical_allow = 0
            for index, (item_attack, item_init, item_site) in enumerate(zip(attack_image, init_images, site)):
                layers1 = self.calc_layers(item_init[0], item_attack[0],
                                           (item_site[0] ** 2 + item_site[1] ** 2) ** 0.5, device)
                layers2 = self.calc_layers(item_init[1], item_attack[1],
                                           (item_site[0] ** 2 + (item_site[1] - 10) ** 2) ** 0.5, device)
                if 0 <= layers1 < 10 and 0 <= layers2 < 10:
                    physical_allow += 1
                if physical_allow > len(attack_image) / 2:
                    best_images = a
        attack_images = best_images
        attack_images = attack_images.cpu()
        attack_images = torch.from_numpy(scaler.inverse_transform(
            attack_images.detach().view(attack_images.shape[0], -1).transpose(0, 1))).transpose(0, 1)
        attack_images = attack_images.to(device)
        images = images.cpu()
        images = torch.from_numpy(scaler.inverse_transform(
            images.detach().view(images.shape[0], -1).transpose(0, 1))).transpose(0, 1)
        images = images.to(device)
        images = images.view(images.shape[0], -1)
        # if not isIncrease:
        #     attack_images = torch.where(attack_images <= images, attack_images, images)
        for index, (item_attack, item_init, item_site) in enumerate(zip(attack_images, init_images, site)):
            layers1 = self.calc_layers(item_init[0], item_attack[0],
                                       (item_site[0] ** 2 + item_site[1] ** 2) ** 0.5, device)
            layers2 = self.calc_layers(item_init[1], item_attack[1],
                                       (item_site[0] ** 2 + (item_site[1] - 10) ** 2) ** 0.5, device)
            best_layer1[index], best_layer2[index] = layers1, layers2
        return attack_images, best_layer1, best_layer2

    def calc_layers(self, init_ap, attack_ap, distance, device):
        gamma = self.calc_gamma(attack_ap - init_ap, distance)
        error = 1e10
        best_layer = 0
        for layers in range(20):
            output = self.physical_model(torch.tensor([float(layers), float(distance)]).to(device))
            if abs(output - gamma) < error:
                error = abs(output - gamma)
                best_layer = layers
        return best_layer

    def calc_gamma(self, delta, distance):
        return delta / 10 / math.log10(distance)
