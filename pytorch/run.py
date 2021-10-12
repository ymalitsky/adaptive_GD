import copy
import numpy as np
import torch

from optimizer import Adsgd
from utils import load_data, accuracy_and_loss, save_results, seed_everything


def run_adgd(net, n_epoch=2, amplifier=0.02, damping=1, weight_decay=0, eps=1e-8, checkpoint=125, batch_size=128, noisy_train_stat=True):
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []
    
    prev_net = copy.deepcopy(net)
    prev_net.to(device)
    net.train()
    prev_net.train()
    lrs = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adsgd(net.parameters(), amplifier=amplifier, damping=damping, weight_decay=weight_decay, eps=eps)
    prev_optimizer = Adsgd(prev_net.parameters(), weight_decay=weight_decay)
            
    for epoch in range(n_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            prev_optimizer.zero_grad(set_to_none=True)

            prev_outputs = prev_net(inputs)
            prev_loss = criterion(prev_outputs, labels)
            prev_loss.backward()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.compute_dif_norms(prev_optimizer)
            prev_net.load_state_dict(net.state_dict())
            optimizer.step()

            running_loss += loss.item()
            if (i % 10) == 0:
                if noisy_train_stat:
                    losses.append(loss.cpu().item())
                    it_train.append(epoch + i * batch_size / N_train)
                lrs.append(optimizer.param_groups[0]['lr'])

            if i % checkpoint == checkpoint - 1:
                if running_loss / checkpoint < 0.01:
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i + 1, running_loss / checkpoint), end='')
                else:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / checkpoint), end='')
                running_loss = 0.0
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                grad_norms.append(np.sum([p.grad.data.norm().item() for p in net.parameters()]))
                net.train()
                it_test.append(epoch + i * batch_size / N_train)
                
        if not noisy_train_stat:
            it_train.append(epoch)
            train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
            train_acc.append(train_a)
            losses.append(train_l)
            net.train()

    del prev_net
    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test), np.array(lrs), np.array(grad_norms))


if __name__ == "__main__":
    # Train ResNet18 on Cifar10 data
    import argparse
    from resnet import ResNet18
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    parser.add_argument('--lr_amplifier', type=float, default=0.02,
        help='Coefficient alpha for multiplying the stepsize by (1+alpha) (default: 0.02).')
    parser.add_argument('--lr_damping', type=float, default=1.,
        help='Divide the inverse smoothness by damping (default: 1.).')
    parser.add_argument('--weight_decay', type=float, default=0.,
        help='Weight decay parameter (default: 0.).')
    parser.add_argument('--batch_size', type=int, default=128,
        help='Number of passes over the data (default: 128).')
    
    parser.add_argument('--n_epoch', type=int, default=120,
        help='Number of passes over the data (default: 120).')
    parser.add_argument('--n_seeds', type=int, default=1,
        help='Number of random seeds to run the method (default: 1).')
    parser.add_argument('--output_folder', type=str, default='./',
        help='Path to the output folder for saving the logs (optional).')
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    N_train = 50000
    trainloader, testloader, num_classes = load_data(batch_size=args.batch_size)
    checkpoint = len(trainloader) // 3 + 1
    amplifier = 0.02
    
    n_seeds = 1
    max_seed = 424242
    rng = np.random.default_rng(42)
    seeds = [rng.choice(max_seed, size=1, replace=False)[0] for _ in range(n_seeds)]

    for r, seed in enumerate(seeds):
        seed_everything(seed)
        net = ResNet18()
        net.to(device)
        losses_adgd, test_losses_adgd, train_acc_adgd, test_acc_adgd, it_train_adgd, it_test_adgd, lrs_adgd, grad_norms_adgd = run_adgd(
            net=net, n_epoch=args.n_epoch, amplifier=args.lr_amplifier, damping=args.lr_damping, weight_decay=args.weight_decay, 
            checkpoint=checkpoint, batch_size=args.batch_size, noisy_train_stat=False
        )
        method = f'adgd_{args.lr_amplifier}_{args.lr_damping}'
        experiment = 'cifar10_resnet18'
        save_results(losses_adgd, test_losses_adgd, train_acc_adgd, test_acc_adgd, it_train_adgd, it_test_adgd, lrs=lrs_adgd, 
                 grad_norms=grad_norms_adgd, method=method, experiment=experiment, folder=args.output_folder)
