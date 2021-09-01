from decolle.lenet_decolle_model import DECOLLELoss, LIFLayer
from decolle.lenet_decolle_model_1D_MN import LenetDECOLLE1DMN, DECOLLELoss, LIFLayer
from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
import os
import numpy as np
import torch
import importlib
import yaml
from tqdm import tqdm
def main():
 
    device = 'cuda'
    np.set_printoptions(precision=4)
    path = '/home/icub/big-data/gridSearchClassMuscWinLenAllSubj/Jun19_21-19-04_iiticubws036'
    args = parse_args(default_params_file=os.path.join(path, 'params.yml'))
    
    with open(args.params_file, 'r') as f:
        params = yaml.load(f)

    dataset = importlib.import_module(params['dataset'])
    
    create_data = dataset.create_dataloader

    ## Load Data

    gen_train, gen_test = create_data(root=params['filename'],
                                        chunk_size_train=params['chunk_size_train'],
                                        chunk_size_test=params['chunk_size_test'],
                                        overlap_size_train_perc=params['overlap_size_train_perc'],
                                        overlap_size_test_perc=params['overlap_size_test_perc'],
                                        perc_test_norm=params['perc_test_norm'],
                                        muscle_to_exclude=params['muscle_to_exclude'],
                                        class_to_include=params['class_to_include'],
                                        thr_firing_excl_slice=params['thr_firing_excl_slice'],
                                        batch_size=1,
                                        dt=params['deltat'],
                                        num_workers=params['num_dl_workers'])

    data_batch, target_batch = next(iter(gen_train))
    data_batch = torch.Tensor(data_batch).to(device)
    target_batch = torch.Tensor(target_batch).to(device)

    #d, t = next(iter(gen_train))
    input_shape = data_batch.shape[-3:]

    #Backward compatibility

    if len(params['input_shape']) == 0:
        params['input_shape'] = [0]

    if params['input_shape'][0] == 0:
        params['input_shape'] = data_batch.shape[-len(params['input_shape']):]

    if type(params['alpha']) is not list:
        params['alpha'] = [params['alpha']]
    if type(params['beta']) is not list:
        params['beta'] = [params['beta']]

    ## Create Model, Optimizer and Loss
    net = LenetDECOLLE1DMN(out_channels=params['out_channels'],
                        Nhid=np.array(params['Nhid']),
                        Mhid=np.array(params['Mhid']),
                        kernel_size=params['kernel_size'],
                        pool_size=params['pool_size'],
                        input_shape=params['input_shape'],
                        alpha=np.array(params['alpha']),
                        alpharp=np.array(params['alpharp']),
                        dropout=[0],
                        beta=np.array(params['beta']),
                        num_conv_layers=params['num_conv_layers'],
                        num_mlp_layers=params['num_mlp_layers'],
                        lc_ampl=params['lc_ampl'],
                        lif_layer_type=LIFLayer,
                        method=params['learning_method'],
                        with_output_layer=True).to(device)
    
    reg_l = params['reg_l'] if 'reg_l' in params else None
    if hasattr(params['learning_rate'], '__len__'):
        from decolle.utils import MultiOpt
        opts = []
        for i in range(len(params['learning_rate'])):
            opts.append(torch.optim.Adamax(net.get_trainable_parameters(i), lr=params['learning_rate'][i], betas=params['betas']))
        opt = MultiOpt(*opts)
    else:
        opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
    if 'loss_scope' in params and params['loss_scope']=='crbp':
        from decolle.lenet_decolle_model import CRBPLoss
        loss = torch.nn.SmoothL1Loss(reduction='none')
        decolle_loss = CRBPLoss(net = net, loss_fn = loss, reg_l=reg_l)
    else:
        loss = [torch.nn.SmoothL1Loss() for i in range(len(net))]
        if net.with_output_layer:
            loss[-1] = cross_entropy_one_hot
        decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=reg_l)

    checkpoint_dir = os.path.join(path, 'checkpoints')

    load_model_from_checkpoint(checkpoint_dir, net, opt, n_checkpoint=-1, device=device)
    
    net.eval()

    with torch.no_grad():
        if hasattr(net.LIF_layers[0], 'base_layer'):
            dtype = net.LIF_layers[0].base_layer.weight.dtype
        else:
            dtype = net.LIF_layers[0].weight.dtype
        device = net.get_input_layer_device()
        iter_data_labels = iter(gen_test)

        for data_batch, target_batch in iter_data_labels:
            data_batch = torch.Tensor(data_batch).type(dtype).to(device)

            timesteps = data_batch.shape[1]

            net.init(data_batch, 0)

            for k in tqdm(range(timesteps)):
                s, r, u = net.forward(data_batch[:, k])     # , :, :])



if __name__ == "__main__":
    main()