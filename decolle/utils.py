import argparse
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime
import os
import socket
import argparse
import math
import yaml
from collections import Counter

def print_params(params):
    print('Using the following parameters:')
    m = max(len(x) for x in params)
    for k, v in zip(params.keys(), params.values()):
        print('{}{} : {}'.format(k, ' ' * (m - len(k)), v))

def parse_args(default_params_file = 'parameters/params.yml'):
    parser = argparse.ArgumentParser(description='DECOLLE for event-driven object recognition')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda)')
    parser.add_argument('--resume_from', type=str, default=None, metavar='path_to_logdir',
                        help='Path to a previously saved checkpoint')
    parser.add_argument('--params_file', type=str, default=default_params_file,
                        help='Path to parameters file to load. Ignored if resuming from checkpoint')
    parser.add_argument('--no_save', dest='no_save', action='store_true',
                        help='Set this flag if you don\'t want to save results')
    parser.add_argument('--save_dir', type=str, default='', help='Name of subdirectory to save results in')
    parser.add_argument('--verbose', type=bool, default=False, help='print verbose outputs')
    
    parsed, unknown = parser.parse_known_args()

    for arg in unknown:
        if arg.startswith(("-", "--")):
            #you can pass any arguments to add_argument
            parser.add_argument(arg, type=str)

    args=parser.parse_args()
    
    if args.no_save:
        print('!!!!WARNING!!!!\n\nRESULTS OF THIS TRAINING WILL NOT BE SAVED\n\n!!!!WARNING!!!!\n\n')

    return args

def prepare_experiment(name, args):
    
    if args.resume_from is None:
        params_file = args.params_file
        if not args.no_save:
            current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join('runs_args_{0}/'.format(name), args.save_dir,
                                   current_time + '_' + socket.gethostname() + '_' +'bioplaus')
            checkpoint_dir = os.path.join(log_dir, 'checkpoints')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            from shutil import copy2
            copy2(params_file, os.path.join(log_dir, 'params.yml'))
            writer = SummaryWriter(log_dir=log_dir)
            print('Saving results to {}'.format(log_dir))
    else:
        log_dir = args.resume_from
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        params_file = os.path.join(log_dir, 'params.yml')
        if not args.no_save:
            writer = SummaryWriter(log_dir=log_dir)
        print('Resuming model from {}'.format(log_dir))

    with open(params_file, 'r') as f:
        params = yaml.load(f)
    directories = {'log_dir':log_dir, 'checkpoint_dir': checkpoint_dir}
    return params, writer, directories

def load_model_from_checkpoint(checkpoint_dir, net, opt, device='cuda'):
    starting_epoch = 0
    checkpoint_list = os.listdir(checkpoint_dir)
    if checkpoint_list:
        checkpoint_list.sort()
        last_checkpoint = checkpoint_list[-1]
        checkpoint = torch.load(os.path.join(checkpoint_dir, last_checkpoint), map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        print('Resuming from epoch {}'.format(starting_epoch))
    return starting_epoch

def train(gen_train, loss, net, opt, epoch, burnin, reg_l):
    device = net.LIF_layers[0].base_layer.weight.device 
    data_batch, target_batch = gen_train.next()
    net.init(data_batch, burnin)
    total_loss = 0
    for k in tqdm(range(data_batch.shape[1]), desc='Epoch {}'.format(epoch)):
        s, r, u = net.forward(torch.Tensor(data_batch[:, k, :, :]).to(device))
        loss_tv = 0
        for i in range(len(net)):
            reg_loss = reg_l[i]*1e-2*torch.mean(torch.relu(u[i]+.01))
            eg2_loss = reg_l[i]*3e-3*(torch.relu(.1-torch.mean(torch.sigmoid(u[i]))))
            loss_tv += loss(r[i], torch.Tensor(target_batch).to(device)) +reg_loss + eg2_loss#+ .1 * torch.mean(s[i])
        loss_tv.backward()
        total_loss += loss_tv.data.detach().cpu().numpy()
        opt.step()
        opt.zero_grad()
    print('Loss {0:1.3}'.format(total_loss))
    return total_loss

def test(gen_test, loss, net, burnin, print_error):
    data_batch, target_batch = gen_test.next()
    net.init(data_batch, burnin)
    device = net.LIF_layers[0].base_layer.weight.device 
    timesteps = data_batch.shape[1]

    r_cum = np.zeros((len(net), timesteps)+ target_batch.shape)
    r_sum = [np.zeros(target_batch.shape)] * len(net)
    s_sum = [0] * len(net)


    print('--------------Testing---------------')
    for k in tqdm(range(timesteps), desc='Testing'):
        s, r, u = net.forward(torch.Tensor(data_batch[:, k, :, :]).to(device))
        r_sum = [l_sum + l.detach().cpu().numpy() for l_sum, l in zip(r_sum, r)]
        s_sum = [l_sum + np.mean(l.detach().cpu().numpy()) for l_sum, l in zip(s_sum, s)]
        r_cum[:, k] = np.array([l.detach().cpu().numpy() for l in r])

    r_sum = [t / timesteps for t in r_sum]
    s_sum = [t / timesteps for t in s_sum]

    test_loss = np.zeros(len(net))
    for i in range(len(net)):
        test_loss[i] = loss(torch.Tensor(r_sum[i]).to(device), torch.Tensor(target_batch).to(device)) + .1 * s_sum[i]

    test_acc = accuracy(r_cum, target_batch)
    if print_error:
        print(' '.join(['Error Rate L{0} {1:1.3}'.format(j, 1-v) for j, v in enumerate(test_acc)]))
    return test_loss, test_acc

def accuracy(output, target):
    maxs = output.argmax(axis=-1)
    targets = target.argmax(axis=-1)
    acc = []
    for m in maxs:
        most_common_out = [Counter(m[:, i]).most_common(1)[0][0] for i in range(m.shape[1])]
        acc.append(np.mean(most_common_out == targets))

    return np.array(acc)


def save_checkpoint(epoch, checkpoint_dir, net, opt):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        }, os.path.join(checkpoint_dir, 'epoch{:05}.tar'.format(epoch)))


def write_stats(epoch, test_acc, test_loss, writer):
    for i, [l, a] in enumerate(zip(test_loss, test_acc)):
        writer.add_scalar('/test_loss/layer{}'.format(i), l, epoch)
        writer.add_scalar('/test_acc/layer{}'.format(i), a, epoch)
        
def get_output_shape(input_shape, kernel_size=[3,3], stride = [1,1], padding=[1,1], dilation=[0,0]):
    if not hasattr(kernel_size, '__len__'):
        kernel_size = [kernel_size, kernel_size]
    if not hasattr(stride, '__len__'):
        stride = [stride, stride]
    if not hasattr(padding, '__len__'):
        padding = [padding, padding]
    if not hasattr(dilation, '__len__'):
        dilation = [dilation, dilation]
    im_height = input_shape[-2]
    im_width = input_shape[-1]
    height = int((im_height + 2 * padding[0] - dilation[0] *
                  (kernel_size[0] - 1) - 1) // stride[0] + 1)
    width = int((im_width + 2 * padding[1] - dilation[1] *
                  (kernel_size[1] - 1) - 1) // stride[1] + 1)
    return [height, width]
