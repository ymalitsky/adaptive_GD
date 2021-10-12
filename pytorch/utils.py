import numpy as np
import os
import random
import torch
import torchvision

import torchvision.transforms as transforms

from pathlib import Path


def seed_everything(seed=1029):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def load_data(dataset='cifar10', batch_size=128, num_workers=4):
    """
    Loads the required dataset
    :param dataset: Can be either 'cifar10' or 'cifar100'
    :param batch_size: The desired batch size
    :return: Tuple (train_loader, test_loader, num_classes)
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Only cifar 10 and cifar 100 are supported')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, num_classes
    
    
def accuracy_and_loss(net, dataloader, device, criterion):
    net.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).cpu().item() / len(dataloader)

    return correct / total, loss

def save_results(losses, test_losses, train_acc, test_acc, it_train, it_test, grad_norms, method='sgd', 
                 lrs=[], experiment='cifar10_resnet18', folder='./', to_save_extra=[], prefixes_extra=[]):
    path = f'./{folder}/{experiment}/'
    Path(path).mkdir(parents=True, exist_ok=True)
    to_save = [losses, test_losses, train_acc, test_acc, it_train, it_test, grad_norms, lrs] + to_save_extra
    prefixes = ['l', 'tl', 'a', 'ta', 'itr', 'ite', 'gn', 'lr'] + prefixes_extra
    for log, prefix in zip(to_save, prefixes):
        np.save(f'{path}/{method}_{prefix}.npy', log)
        
def load_results(method, logs_path, load_lr=False):
    path = logs_path
    if logs_path[-1] != '/':
        path += '/'
    path += method + '_'
    prefixes = ['l', 'tl', 'a', 'ta', 'itr', 'ite', 'gn']
    if load_lr:
        prefixes += ['lr']
    out = [np.load(path + prefix + '.npy') for prefix in prefixes]
    return tuple(out)
