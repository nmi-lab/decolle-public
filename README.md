# Deep Continuous Local Learning (DECOLLE)

DECOLLE is an online learning framework for spiking neural networks. The algorithmic details are described here:

doi:10.3389/fnins.2020.00424

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

All parameter sets are contained in scripts/parameters

## Authors

* **Emre Neftci** - *Initial work* - [eneftci](https://github.com/eneftci)
* **Jacques Kaiser** - [jkaiser](https://github.com/jkaiser)
* **Massi Iacono** - [miacono](https://github.com/miacono)

## License

This project is licensed under the GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details
