import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms, utils
import torch
from torch.utils.data.dataloader import DataLoader
#import tqdm
import copy
from collections import namedtuple

def __gen_ST(N, T, rate, mode = 'regular'):    
    if mode == 'regular':
        spikes = np.zeros([T, N])
        spikes[::(1000//rate)] = 1
        return spikes
    elif mode == 'poisson':
        spikes = np.ones([T, N])        
        spikes[np.random.binomial(1,float(1000. - rate)/1000, size=(T,N)).astype('bool')] = 0
        return spikes
    else:
        raise Exception('mode must be regular or Poisson')
        
def spiketrains(N, T, rates, mode = 'poisson'):
    '''
    *N*: number of neurons
    *T*: number of time steps
    *rates*: vector or firing rates, one per neuron
    *mode*: 'regular' or 'poisson'
    '''
    if not hasattr(rates, '__iter__'):
        return __gen_ST(N, T, rates, mode)
    rates = np.array(rates)
    M = rates.shape[0]
    spikes = np.zeros([T, N])
    for i in range(M):
        if int(rates[i])>0:
            spikes[:,i] = __gen_ST(1, T, int(rates[i]), mode = mode).flatten()
    return spikes

def spikes_to_evlist(spikes):
    t = np.tile(np.arange(spikes.shape[0]), [spikes.shape[1],1])
    n = np.tile(np.arange(spikes.shape[1]), [spikes.shape[0],1]).T  
    return t[spikes.astype('bool').T], n[spikes.astype('bool').T]

def plotLIF(U, S, Vplot = 'all', staggering= 1, ax1=None, ax2=None, **kwargs):
    '''
    This function plots the output of the function LIF.
    
    Inputs:
    *S*: an TxNnp.array, where T are time steps and N are the number of neurons
    *S*: an TxNnp.array of zeros and ones indicating spikes. This is the second
    output return by function LIF
    *Vplot*: A list indicating which neurons' membrane potentials should be 
    plotted. If scalar, the list range(Vplot) are plotted. Default: 'all'
    *staggering*: the amount by which each V trace should be shifted. None
    
    Outputs the figure returned by figure().    
    '''
    V = U
    spikes = S
    #Plot
    t, n = spikes_to_evlist(spikes)
    #f = plt.figure()
    if V is not None and ax1 is None:
        ax1 = plt.subplot(211)
    elif V is None:
        ax1 = plt.axes()
        ax2 = None
    ax1.plot(t, n, 'k|', **kwargs)
    ax1.set_ylim([-1, spikes.shape[1] + 1])
    ax1.set_xlim([0, spikes.shape[0]])

    if V is not None:
        if Vplot == 'all':
            Vplot = range(V.shape[1])
        elif not hasattr(Vplot, '__iter__'):
            Vplot = range(np.minimum(Vplot, V.shape[1]))    
        
        if ax2 is None:
            ax2 = plt.subplot(212)
    
        if V.shape[1]>1:
            for i, idx in enumerate(Vplot):
                ax2.plot(V[:,idx]+i*staggering,'-',  **kwargs)
        else:
            ax2.plot(V[:,0], '-', **kwargs)
            
        if staggering!=0:
            plt.yticks([])
        plt.xlabel('time [ms]')
        plt.ylabel('u [au]')

    ax1.set_ylabel('Neuron ')

    plt.xlim([0, spikes.shape[0]])
    plt.ion()
    plt.show()
    return ax1,ax2




input_shape = [28,28,1]


def to_one_hot(t, width):
    t_onehot = torch.zeros(*t.shape+(width,))
    return t_onehot.scatter_(1, t.unsqueeze(-1), 1)



def image2spiketrain(x,y,gain=50,min_duration=None, max_duration=500):
    y = to_one_hot(y, 10)
    if min_duration is None:
        min_duration = max_duration-1
    batch_size = x.shape[0]
    T = np.random.randint(min_duration,max_duration,batch_size)
    Nin = np.prod(input_shape)
    allinputs = np.zeros([batch_size,max_duration, Nin])
    for i in range(batch_size):
        st = spiketrains(T = T[i], N = Nin, rates=gain*x[i].reshape(-1)).astype(np.float32)
        allinputs[i] =  np.pad(st,((0,max_duration-T[i]),(0,0)),'constant')
    allinputs = np.transpose(allinputs, (1,0,2))
    allinputs = allinputs.reshape(allinputs.shape[0],allinputs.shape[1],1, 28,28)

    alltgt = np.zeros([max_duration, batch_size, 10], dtype=np.float32)
    for i in range(batch_size):
        alltgt[:,i,:] = y[i]

    return allinputs, alltgt

def target_convolve(tgt,alpha=8,alphas=5):
    max_duration = tgt.shape[0]
    kernel_alpha = np.exp(-np.linspace(0,10*alpha,dtype='float')/alpha)
    kernel_alpha /= kernel_alpha.sum()
    kernel_alphas = np.exp(-np.linspace(0,10*alphas,dtype='float')/alphas)
    kernel_alphas /= kernel_alphas.sum()
    tgt = tgt.copy()
    for i in range(tgt.shape[1]):
        for j in range(tgt.shape[2]):
            tmp=np.convolve(np.convolve(tgt[:,i,j],kernel_alpha),kernel_alphas)[:max_duration]
            tgt[:,i,j] = tmp
    return tgt/tgt.max()


datasetConfig = namedtuple('config',['image_size','batch_size','data_path'])

#class DataLoaderIterPreProcessed(_DataLoaderIter):
#    def __next__(self):
#           indices = next(self.sample_iter)  # may raise StopIteration
#           td, tl = self.dataset.data, self.dataset.targets
#           batch = self.collate_fn([(td[i], tl[i]) for i in indices])
#           return batch
#
#class DataLoaderPreProcessed(DataLoader):
#    def __iter__(self):
#        return DataLoaderIterPreProcessed(self)

def sequester(tensor):
    dtype = tensor.dtype
    return torch.tensor(tensor.detach().cpu().numpy(), dtype=dtype)


#def preprocess_dataset(dataset):
#    x, y = dataset.data[0], dataset.targets[0]
#    td = torch.empty(torch.Size([len(dataset)])+x.shape, dtype = torch.float32)
#    if not hasattr(y, 'shape'):
#        tl = torch.empty(torch.Size([len(dataset)]), dtype = torch.int)
#    else:
#        tl = torch.empty(torch.Size([len(dataset)])+y.shape, dtype = y.dtype)
#    for idx in tqdm.tqdm(range(len(dataset)), desc = "Pre-processing dataset"):
#        td[idx], tl[idx] = dataset[idx]
#        
#    dataset.data, dataset.targets = td, tl
#
#
#    dataset.transform = None
#    return dataset


def pixel_permutation(d_size, r_pix=1.0, seed=0):
   import copy
   n_pix = int(r_pix * d_size)
   np.random.seed(seed*1313)
   pix_sel = np.random.choice(d_size, n_pix, replace=False).astype(np.int32)
   pix_prm = np.copy(pix_sel)
   np.random.shuffle(pix_prm)
   perm_inds = np.arange(d_size)
   perm_inds[pix_sel] = perm_inds[pix_prm]
   return perm_inds

def permute_dataset(dataset, r_pix, seed):
    shape = dataset.data.shape[1:]
    datap = dataset.data.view(-1, np.prod(shape)).detach().numpy()
    perm = pixel_permutation(np.prod(shape), r_pix, seed=seed)
    return torch.FloatTensor(datap[:,perm].reshape(-1,*shape))

def partition_dataset(dataset, Nparts=60, part=0):
    N = len(dataset.data)

    idx = np.arange(N, dtype='int')

    step = (N//Nparts)
    idx = idx[step*part:step*(part+1)]

    td = dataset.data[idx]
    tl = dataset.targets[idx]
    return td, tl

def dynaload(dataset,
          batch_size,
          name,
          DL,
          perm=0.,
          Nparts=1,
          part=0,
          seed=0,
          taskid=0,
          base_perm=.0,
          base_seed=0,
          train = True,
          **loader_kwargs):
        if base_perm>0:
            data = permute_dataset(dataset, base_perm, seed=base_seed)
            dataset.data = data
        if perm>0:
            data = permute_dataset(dataset, perm, seed=seed)
            dataset.data = data

        loader = DL(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=dataset.train,
                    **loader_kwargs)

        loader.taskid = taskid
        loader.name = name +'_{}'.format(part)
        loader.short_name = name
        return loader


def mnist_loader_dynamic(
        config,
        train,
        pre_processed = True,
        Nparts=1,
        part=1):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.0,), (1.0,))])


    dataset = datasets.MNIST(root=config.data_path, download=True, transform=transform, train = train)
    if Nparts>1:
        data, targets = partition_dataset(dataset, Nparts, part)
        dataset.data = data
        dataset.targets = targets

    if pre_processed:
        dataset_ = preprocess_dataset(dataset)
        DL = DataLoaderPreProcessed
    else:
        dataset_ = dataset
        DL = DataLoader
    batch_size = config.batch_size
    name = 'MNIST'

    return dataset_, name, DL

def get_mnist_loader(
        batch_size,
        train, 
        perm=0.,
        Nparts=1,
        part=0,
        seed=0,
        taskid=0,
        pre_processed=False,
        base_perm=.0,
        base_seed=0,
        **loader_kwargs):
    
    config = datasetConfig(image_size = [28,28], batch_size = batch_size, data_path = './data/mnist')
    
    d,name,dl = mnist_loader_dynamic(config, train, pre_processed, Nparts, part)
    
    return dynaload(
          d,
          config.batch_size,
          name,
          dl,
          perm=perm,
          Nparts=Nparts,
          part=part,
          seed=seed,
          taskid=taskid,
          base_perm=base_perm,
          base_seed=base_seed,
          train = train,
          **loader_kwargs)

def usps_loader_dynamic(config, train, pre_processed=False, Nparts=1, part=1):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    from usps_loader import USPS
    
    transform = transforms.Compose([
#                    transforms.ToPILImage(),
                    transforms.Resize(config.image_size),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.0,), (1.0,))])
    
    dataset = USPS(root=config.data_path, download=True, transform=transform, train = train)
    name = 'USPS'

    if Nparts>1:
        partition_dataset(dataset, Nparts, part)

    if pre_processed:
        dataset = preprocess_dataset(dataset)
        DL = DataLoaderPreProcessed
    else:
        DL = DataLoader

    return dataset,name, DL


def get_usps_loader(config, train, perm=0., Nparts=1, part=0, seed=0, taskid=0, pre_processed=False, **loader_kwargs):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    dataset,name,DL = usps_loader_dynamic(config, train, pre_processed)
    
    return dynaload(dataset,
          config.batch_size,
          name,
          DL,
          perm=perm,
          Nparts=Nparts,
          part=part,
          seed=seed,
          taskid=taskid,
          base_perm=base_perm,
          base_seed=base_seed,
          train = train,
          **loader_kwargs)

def svhn_loader_dynamic(config, train, pre_processed=False, Nparts=1, part=1):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    transform = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.35,), (0.65,)),
                    transforms.Lambda(lambda x: x.view(np.prod(config.image_size))),
                    transforms.Lambda(lambda x: x*2-1)])
    
    name = 'SVHN'
    
    dataset = datasets.SVHN(root=config.data_path, download=True, transform=transform, split = 'train' if train else 'test')
    dataset.train = train

    if Nparts>1:
        partition_dataset(dataset, Nparts, part)

    if pre_processed:
        dataset = preprocess_dataset(dataset)
        DL = DataLoaderPreProcessed
    else:
        DL = DataLoader

    return dataset,name, DL

def get_svhn_loader_dynamic(config, train, perm=0, taskid=0, seed=0, Nparts=1, part=0,  pre_processed=False, **loader_kwargs):
    dataset,name,DL = svhn_loader_dynamic(config, train, pre_processed)
    
    return dynaload(dataset,
          config.batch_size,
          name,
          DL,
          perm=perm,
          Nparts=Nparts,
          part=part,
          seed=seed,
          taskid=taskid,
          base_perm=base_perm,
          base_seed=base_seed,
          train = train,
          **loader_kwargs)
