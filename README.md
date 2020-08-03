# Deep Continuous Local Learning (DECOLLE)

DECOLLE is an online learning framework for spiking neural networks.
The algorithmic details are described in this [Frontiers paper](https://www.frontiersin.org/articles/10.3389/fnins.2020.00424/full).
If you use this work in your research, please cite as:

```
@ARTICLE{decolle2020,
AUTHOR={Kaiser, Jacques and Mostafa, Hesham and Neftci, Emre},
TITLE={Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)},
JOURNAL={Frontiers in Neuroscience},
VOLUME={14},
PAGES={424},
YEAR={2020},
URL={https://www.frontiersin.org/article/10.3389/fnins.2020.00424},
DOI={10.3389/fnins.2020.00424},
ISSN={1662-453X}
```

### Installing
Clone and install. The Python setuptools will take care of dependencies
```
git clone https://github.com/nmi-lab/decolle-public.git
cd decolle-public
python setup.py install --user
```

The following will run decolle on the default parameter set
```
cd scripts
python train_lenet_decolle.py
```

All parameter sets are contained in scripts/parameters, you can use them as such:
```
cd scripts
python train_lenet_decolle.py --params_file=parameters/params_dvsgestures_torchneuromorphic_attention.yml
```

## Authors

* **Emre Neftci** - *Initial work* - [eneftci](https://github.com/eneftci)
* **Jacques Kaiser** - [jackokaiser](https://github.com/jackokaiser)
* **Massi Iacono** - [miacono](https://github.com/miacono)

## License

This project is licensed under the GPLv3 License - see the [LICENSE.txt](LICENSE.txt) file for details
