import math

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
import csv

from torch_npu.npu import amp
from torch_npu.contrib import transfer_to_npu

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
        self.scaler = amp.GradScaler()
        

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
            learning_rate = 0.06
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
            w = torch.torch.full_like(images, -1).to(device)
        w.requires_grad = True
        w_lower_bound = 1 / 2 * torch.log(-(images_lower_bound / (images_lower_bound - 1)))
        w_lower_bound = w_lower_bound.to(device)
        w_upper_bound = 1 / 2 * torch.log(-(images_upper_bound / (images_upper_bound - 1)))
        w_upper_bound = torch.where(torch.isnan(w_upper_bound), torch.inf, w_upper_bound)
        w_upper_bound = w_upper_bound.to(device)

        optimizer = optim.Adam([w], lr=learning_rate)
        for step in range(max_iter):
            a = 1 / 2 * (nn.Tanh()(w) + 1)
            a = a.to(device)
            # 第一个目标，对抗样本与原始样本足够接近
            loss1 = None
            mask = a > images
            lambda_ = 0.003
            mask = mask.float() + lambda_ * (~mask).float()
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
                    loss1 = torch.sum(torch.abs(mask * (images - a)))
            elif mode == 'L2':
                if isIncrease:
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
                    loss1 = torch.max(torch.abs(mask * (a - images)))

            # 第二个目标，误导模型输出
            loss2 = torch.sum(c * f(mask * a))
            cost = loss1 + loss2
            optimizer.zero_grad()
            self.scaler.scale(cost).backward() 
            self.scaler.step(optimizer)
            self.scaler.update()
            # cost.backward()
            # optimizer.step()

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


class AutoAttack:
    def __init__(self, model, device):
        self.device = device
        self.model = model

    def generate_data(self, data):
        loss_function = nn.MSELoss()
        x_adv_pgd, y = pgd_attack(self.model, data[0].to(self.device), data[1].to(self.device), epsilon=10, alpha=0.1, num_iter=100)
        x_adv_fab, y = fab_attack(self.model, data[0].to(self.device), data[1].to(self.device), epsilon=10, alpha=0.1, num_iter=100)
        x_adv_square, y = square_attack(self.model, data[0].to(self.device), data[1].to(self.device), epsilon=10, num_iter=100)
        outputs1, outputs2 = self.model(x_adv_pgd)
        loss_pgd = loss_function(outputs1.view(-1).float(), y[:, 0].float()) + loss_function(outputs2.view(-1).float(),
                                                                                             y[:, 1].float())
        outputs1, outputs2 = self.model(x_adv_pgd)
        loss_fab = loss_function(outputs1.view(-1).float(), y[:, 0].float()) + loss_function(outputs2.view(-1).float(),
                                                                                             y[:, 1].float())
        outputs1, outputs2 = self.model(x_adv_square)
        loss_square = loss_function(outputs1.view(-1).float(), y[:, 0].float()) + loss_function(outputs2.view(-1).float(),
                                                                                             y[:, 1].float())
        res = None
        if loss_pgd >= loss_fab and loss_pgd >= loss_square:
            res = x_adv_pgd
        if loss_fab >= loss_pgd and loss_fab >= loss_square:
            res = x_adv_fab
        if loss_square >= loss_fab and loss_square >= loss_pgd:
            res = x_adv_square
        
        return res, y

def pgd_attack(model, images, y, epsilon, alpha, num_iter):

    scaler = amp.GradScaler()
    # 初始化对抗样本
    images = images.view(-1, 1, len(images[0]))
    original_images = images.clone().detach()
    adv_images = original_images + torch.empty_like(original_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, -100, 0)

    for _ in range(num_iter):
        adv_images.requires_grad = True
        outputs1, outputs2 = model(adv_images)
        
        # 计算损失
        loss_function = nn.MSELoss()
        loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) + loss_function(outputs2.view(-1).float(),
                                                                                             y[:, 1].float())
        scaler.scale(loss).backward() 
        grad_sign = adv_images.grad.data.sign()
        
        # 梯度上升更新
        adv_images = adv_images + alpha * grad_sign
        
        # 投影到原始样本的epsilon邻域内
        eta = torch.clamp(adv_images - original_images, -epsilon, epsilon)
        adv_images = torch.clamp(original_images + eta, -100, 0).detach_()
    
    return adv_images, y

def fab_attack(model, images, y, epsilon, alpha, num_iter):
    scaler = amp.GradScaler()
    images = images.view(-1, 1, len(images[0]))
    original_images = images.clone().detach()
    adv_images = images.clone().detach()
    batch_size = images.size(0)
    
    # 初始化步长
    alpha = alpha * torch.ones(batch_size).to(images.device)
    
    for i in range(num_iter):
        adv_images.requires_grad = True
        outputs1, outputs2 = model(adv_images)
        
        loss_function = nn.MSELoss()
        # 计算损失，目标是最小化正确类别的置信度
        loss = loss_function(outputs1.view(-1).float(), y[:, 0].float()) + loss_function(outputs2.view(-1).float(),
                                                                                             y[:, 1].float())
        model.zero_grad()
        scaler.scale(loss).backward() 
        
        # 计算梯度
        grad_sign = adv_images.grad.data.sign()
        
        # 计算扰动
        with torch.no_grad():
            perturb = alpha.view(-1, 1, 1) * grad_sign
            
            # 更新对抗样本
            adv_images = adv_images - perturb
            
            # 投影到epsilon范围内
            eta = torch.clamp(adv_images - original_images, -epsilon, epsilon)
            adv_images = torch.clamp(original_images + eta, -100, 0)
        
        # 动态调整步长 alpha
        with torch.no_grad():
            new_outputs1, new_outputs2 = model(adv_images)
            new_loss = torch.sqrt((new_outputs1.view(-1) - y[:, 0]) ** 2 + (new_outputs2.view(-1) - y[:, 1]) ** 2)
            old_loss = torch.sqrt((outputs1.view(-1) - y[:, 0]) ** 2 + (outputs2.view(-1) - y[:, 1]) ** 2)
            success = new_loss > old_loss
            alpha[success] = alpha[success] * 0.9
            alpha[~success] = alpha[~success] * 1.1
    
    return adv_images, y

def square_attack(model, images, y, epsilon, num_iter, p_init=0.05):
    scaler = amp.GradScaler()
    images = images.view(-1, 1, len(images[0]))
    adv_images = images.clone().detach()
    batch_size, h, w = images.shape

    for i in range(num_iter):
        # 随机选择位置和方块大小
        for j in range(batch_size):
            h_size = int(np.round(np.sqrt(p_init) * h))
            w_size = int(np.round(np.sqrt(p_init) * w))
            h_start = np.random.randint(0, h - h_size)
            w_start = np.random.randint(0, w - w_size)
            
            # 保存原始方块
            original_patch = adv_images[j, h_start:h_start + h_size, w_start:w_start + w_size].clone()
            
            # 添加随机噪声方块
            adv_images[j, h_start:h_start + h_size, w_start:w_start + w_size] = \
                torch.clamp(adv_images[j, h_start:h_start + h_size, w_start:w_start + w_size] + \
                            torch.empty(h_size, w_size).uniform_(-epsilon, epsilon).to(images.device), 0, 1)
        
        # 评估新的对抗样本
        with torch.no_grad():
            new_outputs1, new_outputs2 = model(adv_images)
            outputs1, outputs2 = model(images)
        
        # 计算差异
        new_loss = torch.sqrt((new_outputs1.view(-1) - y[:, 0]) ** 2 + (new_outputs2.view(-1) - y[:, 1]) ** 2)
        original_loss = torch.sqrt((outputs1.view(-1) - y[:, 0]) ** 2 + (outputs2.view(-1) - y[:, 1]) ** 2)
        
        # 如果新的扰动效果更差，恢复原始方块
        for j in range(batch_size):
            if new_loss[j].mean() < original_loss[j].mean():
                adv_images[j, h_start:h_start + h_size, w_start:w_start + w_size] = original_patch
    
    return adv_images, y