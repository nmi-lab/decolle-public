#!/bin/python
#-----------------------------------------------------------------------------
# File Name : allconv_decolle.py
# Author: Emre Neftci
#
# Creation Date : Wed 07 Aug 2019 07:00:31 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from .base_model import *
import torch.utils.data
import torchvision

def one_hot(y, out_channels = 10):
    y_onehot = torch.FloatTensor(y.shape[0], out_channels)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot

class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

class LenetDELLE(DECOLLEBase):
    def __init__(self,
                 input_shape,
                 Nhid=[1],
                 Mhid=[128],
                 out_channels=1,
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 dropout=[0.5],
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 lc_ampl=.5):

        num_layers = num_conv_layers + num_mlp_layers
        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:   kernel_size = kernel_size * num_conv_layers
        if len(stride) == 1:        stride = stride * num_conv_layers
        if len(pool_size) == 1:     pool_size = pool_size * num_conv_layers
        if len(dropout) == 1:       self.dropout = dropout = dropout * num_layers
        if len(Nhid) == 1:          self.Nhid = Nhid = Nhid * num_conv_layers
        if len(Mhid) == 1:          self.Mhid = Mhid = Mhid * num_mlp_layers

        super(LenetDELLE, self).__init__()

        # Computing padding to preserve feature size
        padding = (np.array(kernel_size) - 1) // 2  # TODO try to remove padding

        feature_height = input_shape[1]
        feature_width = input_shape[2]

        # THe following lists need to be nn.ModuleList in order for pytorch to properly load and save the state_dict
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        Nhid = [input_shape[0]] + Nhid
        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers

        for i in range(num_conv_layers):
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]

            layer = nn.Conv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i]).to(device)

            pool = nn.MaxPool2d(kernel_size=pool_size[i]).to(device)

            readout = nn.Linear(int(feature_height * feature_width * Nhid[i + 1]), out_channels).to(device)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            dropout_layer = nn.Dropout(dropout[i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

        mlp_in = int(feature_height * feature_width * Nhid[-1])
        Mhid = [mlp_in] + Mhid
        for i in range(num_mlp_layers):
            layer = nn.Linear(Mhid[i], Mhid[i+1]).to(device)
            readout = nn.Linear(Mhid[i+1], out_channels).to(device)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            dropout_layer = nn.Dropout(dropout[self.num_conv_layers+i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)



    def forward(self, input):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers):
            if i == self.num_conv_layers: 
                input = input.view(input.size(0), -1)
            u = lif(input)
            u_p = pool(u)
            u_sig = do(torch.sigmoid(u_p))
            r = ro(u_sig.reshape(u_sig.size(0), -1))
            #non-linearity must be applied after the pooling to ensure that the same winner is selected
            s_out.append(torch.sigmoid(u_p)) 
            r_out.append(r)
            u_out.append(u_p)
            input = torch.sigmoid(u_p.detach())
            i+=1

        return s_out, r_out, u_out

def create_data(batch_size_train=32, batch_size_test=32):
    train_loader = torch.utils.data.DataLoader(
                           torchvision.datasets.MNIST('datamnist/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
                           batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                           torchvision.datasets.MNIST('datamnist/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
                           batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader



def get_optimizer(net, params):
    if params['optimizer'] == 'adam':
        opt = optim.Adam(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
    elif params['optimizer'] == 'adamax':
        opt = optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
    elif params['optimizer'] == 'adagrad':
        opt = optim.Adagrad(net.get_trainable_parameters(), lr=params['learning_rate'])
    else:
        raise ValueError('Please provide a valid entry for optimizer. Possible choices are:\nadam\nadamax')
    return opt

def parse_args():
    parser = argparse.ArgumentParser(description='DELLE for event-driven object recognition')
    parser.add_argument('--resume_from', type=str, default=None, metavar='path_to_logdir',
                        help='Path to a previously saved checkpoint')
    parser.add_argument('--params_file', type=str, default='parameters/params_delle.yml',
                        help='Path to parameters file to load. Ignored if resuming from checkpoint')
    parser.add_argument('--no_save', dest='no_save', action='store_true',
                        help='Set this flag if you don\'t want to save results')
    parser.add_argument('--save_dir', type=str, default='', help='Name of subdirectory to save results in')
    args = parser.parse_args()

    if args.no_save:
        print('!!!!WARNING!!!!\n\nRESULTS OF THIS TRAINING WILL NOT BE SAVED\n\n!!!!WARNING!!!!\n\n')
    print('Running on {}\n'.format(device))

    return args


def train(gen_train, loss, net, opt, epoch, burnin, regularize):
    data_batch, target_batch = iter(gen_train).next()
    data_batch = data_batch.unsqueeze(1)
    if loss.do_onehot:
        target_batch = one_hot(target_batch, 10)
    reg_l = [0,0, .5] #strength of regularization per layer
    for k in range(data_batch.shape[1]):
        d = data_batch[:, k, :, :]
        s, r, u = net.forward(torch.Tensor(d).to(device))
        loss_tv = 0
        for i in range(len(net)):
            #reg_loss = reg_l[i]*1e-2*regularize*torch.mean(torch.relu(u[i]+.01))
            #eg2_loss = reg_l[i]*3e-3*regularize*(torch.relu(.1-torch.mean(torch.sigmoid(u[i]))))
            loss_tv += loss(r[i], target_batch.to(device)) #+reg_loss + eg2_loss#+ .1 * torch.mean(s[i])
        loss_tv.backward()
        opt.step()
        opt.zero_grad()
    return loss_tv

def test(gen_test, loss, net, burnin, out_channels):
    net.eval()
    data_batch, target_batch = iter(gen_test).next()
    data_batch = data_batch.unsqueeze(1)
    if loss.do_onehot:
        target_batch = one_hot(target_batch, 10)
        
    timesteps = data_batch.shape[1]

    r_cum = np.zeros((len(net), timesteps)+ (target_batch.shape[0],out_channels))
    r_sum = [np.zeros([target_batch.shape[0],out_channels])] * len(net)
    s_sum = [0] * len(net)


    for k in tqdm(range(timesteps), desc='Testing'):
        s, r, u = net.forward(torch.Tensor(data_batch[:, k, :, :]).to(device))
        r_sum = [l_sum + l.detach().cpu().numpy() for l_sum, l in zip(r_sum, r)]
        s_sum = [l_sum + np.mean(l.detach().cpu().numpy()) for l_sum, l in zip(s_sum, s)]
        r_cum[:, k] = np.array([l.detach().cpu().numpy() for l in r])

    r_sum = [t / timesteps for t in r_sum]
    s_sum = [t / timesteps for t in s_sum]

    test_loss = np.zeros(len(net))
    for i in range(len(net)):
        test_loss[i] = loss(torch.Tensor(r_sum[i]).to(device), target_batch.to(device)) 
    if not loss.do_onehot:
        tb = one_hot(target_batch, out_channels)
    else:
        tb = target_batch
    test_acc = accuracy(r_cum, tb.detach().cpu().numpy())

    net.train()
    return test_loss, test_acc


if __name__ == "__main__":

    args = parse_args()

    starting_epoch = 0

    # Initializing folders to save results and/or resume training from
    if args.resume_from is None:
        params_file = args.params_file
        if not args.no_save:
            current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join('runs_args_delle/', args.save_dir,
                                   current_time + '_' + socket.gethostname())
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

    num_layers = params['num_conv_layers'] +  params['num_mlp_layers']
    gen_train, gen_test = create_data(params['batch_size'],params['batch_size_test'])

    net = LenetDELLE(out_channels=params['out_channels'],
               Nhid=params['Nhid'],
               Mhid=params['Mhid'],
               kernel_size=params['kernel_size'],
               pool_size=params['pool_size'],
               input_shape=params['input_shape'],
               num_conv_layers=params['num_conv_layers'],
               num_mlp_layers=params['num_mlp_layers'],
               lc_ampl=params['lc_ampl']).to(device)

    print("built")
    loss = eval(params['loss']+"()")
    if  type(loss) == type(nn.CrossEntropyLoss()):
        loss.do_onehot = False
    else:
        loss.do_onehot = True

    opt = get_optimizer(net, params)

    if args.resume_from is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        net.init(iter(gen_train).next()[0].unsqueeze(1), num_layers)
        starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt)
        print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))
    #net.init_parameters(iter(gen_train).next()[0].unsqueeze(1))

    # Printing parameters
    print('Using the following parameters:')
    m = max(len(x) for x in params)
    for k, v in zip(params.keys(), params.values()):
        print('{}{} : {}'.format(k, ' ' * (m - len(k)), v))

    print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))

    # --------TRAINING LOOP----------
    for e in range(starting_epoch , params['num_epochs'] ):
        interval = e // params['lr_drop_interval']
        lr = opt.param_groups[-1]['lr']
        if interval > 0:
            opt.param_groups[-1]['lr'] = params['learning_rate'] / (interval * params['lr_drop_factor'])
        else:
            opt.param_groups[-1]['lr'] = params['learning_rate']
        if lr != opt.param_groups[-1]['lr']:
            print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))

        if e == 1 or (e % params['test_interval']) == 0:
            if not args.no_save:
                print('---------Saving checkpoint---------')
                save_checkpoint(e, checkpoint_dir, net, opt)

            print('--------------Testing---------------')
            test_loss, test_acc = test(gen_test, loss, net, params['burnin_steps'], params['out_channels'])

            print('---------------Epoch {}-------------'.format(e))
            print(' '.join(['L{0} {1:1.3}'.format(j, v) for j, v in enumerate(test_acc)]))
            
            if not args.no_save:
                write_stats(e, test_acc, test_loss, writer)
                writer.add_scalar('/learning_rate', (opt.param_groups[-1]['lr']), e)

            print('---------Resuming training----------')

        train(gen_train, loss, net, opt, e, params['burnin_steps'], regularize = True)



