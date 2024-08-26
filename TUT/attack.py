import math

import numpy as np
import torch
import torchattacks
from sklearn.preprocessing import MinMaxScaler
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
import csv


class CWAttack:
    def __init__(self, model, device, c=0.1, kappa=0, max_iter=100, learning_rate=0.01, mode='L2', isIncrease=False,
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

    def generate_data(self, data):
        """

        :param data: DataLoader
        :return: DataLoader being attacked
        """

        x_attack = self.cw_l2_attack(model=self.model, images=data[0], labels=data[1], device=self.device, c=self.c,
                                     kappa=self.kappa, max_iter=self.max_iter, learning_rate=self.learning_rate,
                                     mode=self.mode, isIncrease=self.isIncrease, bias=self.bias)
        # same = x_attack == data[0]
        # print("\n{} elements changed".fomat(np.count_nonzero(same == False)))
        return x_attack.float(), data[1]

    def cw_l2_attack(self, model, images, labels, device, targeted=False, c=0.1, kappa=2, max_iter=100,
                     learning_rate=0.01, mode='L2', isIncrease=True, bias=0.05):

        class CustomMSELoss(nn.Module):
            def __init__(self):
                super(CustomMSELoss, self).__init__()

            def forward(self, a, images, mask):
                loss = ((mask * (a - images)) ** 2).sum()
                return loss

        # Define f-function
        def f(x):
            outputs1, outputs2 = model(x)
            return -torch.sqrt((outputs1.view(-1) - labels[:, 0]) ** 2 + (outputs2.view(-1) - labels[:, 1]) ** 2)

        assert mode in ['L0', 'L2', 'Linf']
        if mode == 'Linf':
            learning_rate = 0.1
            max_iter = 200

        images_lower_bound = (images - torch.abs(images) * bias).clamp(torch.min(images), torch.max(images))
        images_upper_bound = (images + torch.abs(images) * bias).clamp(torch.min(images), torch.max(images))

        scaler = MinMaxScaler(feature_range=(0, 1))
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

        w = torch.torch.full_like(images, 0).to(device)
        if not isIncrease:
            w = torch.torch.full_like(images, -0.8).to(device)
        w.requires_grad = True

        optimizer = optim.Adam([w], lr=learning_rate)
        for step in range(max_iter):
            # if step > max_iter / 2 and not(not isIncrease and mode == 'L2'):；
            #     w.data = torch.max(torch.min(w, w_upper_bound), w_lower_bound)
            a = 1 / 2 * (nn.Tanh()(w) + 1)
            a = a.to(device)
            # 第一个目标，对抗样本与原始样本足够接近
            loss1 = None
            mask = a > images
            mask = mask.float() + 0.01 * (~mask).float()
            bias_mask = torch.logical_and(a < images_upper_bound, a > images_lower_bound)
            bias_mask_not = torch.logical_not(bias_mask)

            if mode == 'L0':
                if isIncrease:
                    if step > max_iter * 0.8:
                        loss1 = 0.1 * torch.sum(torch.abs(bias_mask * images - bias_mask * a))
                        loss1 = loss1 + torch.sum(torch.abs(bias_mask_not * images - bias_mask_not * a))
                    else:
                        loss1 = torch.sum(torch.abs(images - a))
                else:
                    loss1 = torch.norm(mask * (images - a), p=0)
            elif mode == 'L2':
                if isIncrease:
                    loss1 = None
                    if step > max_iter * 0.8:
                        loss1 = 0.1 * torch.nn.MSELoss(reduction='sum')(bias_mask * a, bias_mask * images)
                        loss1 = loss1 + torch.nn.MSELoss(reduction='sum')(bias_mask_not * a, bias_mask_not * images)
                    else:
                        loss1 = torch.nn.MSELoss(reduction='sum')(a, images)
                else:
                    loss1 = CustomMSELoss()(a, images, mask)
            elif mode == 'Linf':
                if isIncrease:
                    if step > max_iter * 0.8:
                        loss1 = 0.1 * torch.max(torch.abs(bias_mask * a - bias_mask * images))
                        loss1 = loss1 + torch.max(torch.abs(bias_mask_not * a - bias_mask_not * images))
                    else:
                        loss1 = torch.max(torch.abs(a - images))
                else:
                    if step > max_iter * 0.8:
                        loss1 = torch.max(torch.abs(mask * (a - images)))
                    else:
                        loss1 = torch.max(torch.abs(a - images))


            # 第二个目标，误导模型输出
            loss2 = torch.sum(c * f(a))
            cost = loss1 + loss2
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            # if step % (max_iter // 10) == 0:
            #     if cost > prev:
            #         print('Attack Stopped due to CONVERGENCE....')
            #         return a
            #     prev = cost

        attack_images = 1 / 2 * (nn.Tanh()(w) + 1)
        higher = attack_images > images_upper_bound
        lower = attack_images < images_lower_bound
        bias_mask = torch.logical_and(attack_images < images_upper_bound, attack_images > images_lower_bound)
        if isIncrease:
            attack_images = higher * images_upper_bound + lower * images_lower_bound + bias_mask * attack_images
        attack_images = attack_images.cpu()
        attack_images = torch.from_numpy(scaler.inverse_transform(
            attack_images.detach().view(attack_images.shape[0], -1).transpose(0, 1))).transpose(0, 1)
        attack_images = attack_images.to(device)
        images = images.cpu()
        images = torch.from_numpy(scaler.inverse_transform(
            images.detach().view(images.shape[0], -1).transpose(0, 1))).transpose(0, 1)
        images = images.to(device)
        images = images.view(images.shape[0], -1)
        if not isIncrease:
            attack_images = torch.where(attack_images <= images, attack_images, images)
        return attack_images


class FGSMAttack:
    def __init__(self, epsilon=0.05, device='cpu'):
        self.epsilon = epsilon
        self.device = device

    def generate_data(self, images, grad_sign):
        scaler = MinMaxScaler(feature_range=(0, 1))
        images = images.cpu()
        images = torch.from_numpy(scaler.fit_transform(images.detach().numpy().transpose(0, 1)))
        attack_images = images + self.epsilon * grad_sign
        attack_images = torch.clamp(attack_images, 0, 1)
        attack_images = attack_images.cpu()
        attack_images = torch.from_numpy(scaler.inverse_transform(attack_images.detach()))
        return attack_images
