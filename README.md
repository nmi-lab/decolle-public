# A spiking neural network for processing spiking neural data

We adapted the DECOLLE framework to process spiking neural activity.

In particular, we processed the activity of human spinal motor neurons for movement intention recognition. 
This framework is integrated in a non-invasive interface that decodes the activity of motor neurons innervating intrinsic and extrinsic hand muscles. 
In fact, one of the main limitations of current neural interfaces is that machine learning models cannot exploit the efficiency of the spike encoding operated by the nervous system. Spiking-based pattern recognition would detect the spatio-temporal sparse activity of a neuronal pool and lead to adaptive and compact implementations, eventually running locally in embedded systems.
Emergent Spiking Neural Networks (SNN) have not yet been used for processing the activity of in-vivo human neurons.

* **Simone Tanzarella** * - [Tanza13](https://github.com/Tanza13)
* **Massi Iacono** - [miacono](https://github.com/miacono)

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
