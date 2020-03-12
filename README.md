# Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)

This repo contains the [PyTorch](https://pytorch.org/) implementation of the DECOLLE learning rule presented in [this paper](https://arxiv.org/abs/1811.10766).
If you use this code in a scientific publication, please include the following reference in your bibliography:

```
@article{kaiser2018synaptic,
  title={Synaptic Plasticity Dynamics for Deep Continuous Local Learning},
  author={Kaiser, Jacques and Mostafa, Hesham and Neftci, Emre},
  journal={arXiv preprint arXiv:1811.10766},
  year={2018}
}
```

## Install

This repo is a python package depending on [PyTorch](https://pytorch.org/).
You can install it in a virtual environment (or locally with `--user`) with the following command:

```bash
pip install -e .
pip install -r requirements.txt
```

By using the `-e` option of pip, the files will be symlink'ed to your virtualenv instead of copied.
This means that you can modify the files of this repo without having to install it again for the changes to take effect.

## Run

You can reproduce the results presented in the [original paper](https://arxiv.org/abs/1811.10766) by running:

```bash
python scripts/train_lenet_decolle.py --params_file=scripts/parameters/params_nmnist.yml
python scripts/train_lenet_decolle.py --params_file=scripts/parameters/params_dvsgesture.yml
```

The dataset will be automatically downloaded into the `data` folder.
This depends on the [torchneuromorphic](https://github.com/nmi-lab/torchneuromorphic) python package which should be automatically installed from the `requirements.txt`.
