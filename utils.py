
import torch
import torch.nn as nn
import torchvision
import tqdm
import torchvision.transforms as transforms

import os


import numpy as np

# from rdp_accountant import compute_rdp, get_privacy_spent
from opacus.privacy_analysis import compute_rdp, get_privacy_spent


def get_data_loader(dataset, batchsize, augmentation_false):
    if(dataset == 'svhn'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.SVHN('./data', split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=73257, shuffle=True, num_workers=0) #load full btach into memory, to concatenate with extra data

        extraset = torchvision.datasets.SVHN('./data', split='extra', download=True, transform=transform)
        extraloader = torch.utils.data.DataLoader(extraset, batch_size=531131, shuffle=True, num_workers=0) #load full btach into memory

        testset = torchvision.datasets.SVHN('./data', split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)
        return trainloader, extraloader, testloader, len(trainset)+len(extraset), len(testset)
    else:
        if augmentation_false:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
        trainset = CachedCIFAR10(root='./data', train=True, download=True, wrapper_transform=transform_train)
        testset = CachedCIFAR10(root='./data', train=False, download=True, wrapper_transform=transform_test)
        # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)
        t_trainloader = tqdm.tqdm(trainloader)
        for _ in t_trainloader:
            t_trainloader.set_description(f'Cached training data: {len(trainloader.dataset.cached_data)}')
        t_testloader = tqdm.tqdm(testloader)
        for _ in t_testloader:
            t_testloader.set_description(f'Cached training data: {len(testloader.dataset.cached_data)}')
        trainloader.dataset.set_use_cache(use_cache=True)
        trainloader.num_workers = 2
        testloader.dataset.set_use_cache(use_cache=True)
        testloader.num_workers = 2
        return trainloader, testloader, len(trainset), len(testset)


class CachedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download, wrapper_transform, use_cache=False):
        super(CachedCIFAR10, self).__init__(root=root, train=train, transform=None, download=download)
        self.cached_data = []
        self.cached_target = []
        self.use_cache = use_cache
        self.wrapper_transform = wrapper_transform

    def __getitem__(self, index):
        if not self.use_cache:
            img, label = super(CachedCIFAR10, self).__getitem__(index)
            self.cached_data.append(img)
            self.cached_target.append(label)
        else:
            img, label = self.cached_data[index], self.cached_target[index]
        if self.wrapper_transform is not None:
            img = self.wrapper_transform(img)
        return img, label

    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = np.stack(self.cached_data, axis=0)
            self.cached_target = np.stack(self.cached_target, axis=0)
        else:
            self.cached_data = []
            self.cached_target = []
        self.use_cache = use_cache


def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if(rgp):
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _ = get_privacy_spent(orders, rdp, delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break    
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    cur_sigma = init_sigma
    
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps


def restore_param(cur_state, state_dict):
    own_state = cur_state
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)

def sum_list_tensor(tensor_list, dim=0):
    return torch.sum(torch.cat(tensor_list, dim=dim), dim=dim)

def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def checkpoint(net, acc, epoch, sess):
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'approx_error': net.gep.approx_error
    }
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + sess  + '.ckpt')

def adjust_learning_rate(optimizer, init_lr, epoch, all_epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    decay = 1.0
    if(epoch<all_epoch*0.5):
        decay = 1.
    elif(epoch<all_epoch*0.75):
        decay = 10.
    else:
        decay = 100.

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / decay
    return init_lr / decay
