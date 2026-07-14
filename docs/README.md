<div align="center">    
 
# MaleficNet:     
## Hiding Malware into Deep Neural Networks using Spread-Spectrum Channel Coding
[![Conference](https://img.shields.io/badge/esorics-2022-red)](https://esorics2022.compute.dtu.dk/)  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) 

</div>

This is the *demo* of `MaleficNet,` a new spread-spectrum-based technique to hide malware into Deep Neural Networks (DNN). To test the *demo* you need to put the payload (e.g., downloaded from [TheZoo](https://github.com/ytisf/theZoo)), inside the `payload/` folder and follow the instructions below. 

**DISCLAIMER:** the samples from [TheZoo](https://github.com/ytisf/theZoo) are **live** and **dangerous** malware. Do **NOT** download them and run this *demo* unless you are absolutely sure of what you are doing! They are to be used only for **educational purposes**. We highlight that since this demo is for **educational purposes**, we included *only* the injection and extraction algorithms on a single DNN architecture/task (i.e., DenseNet and CIFAR10).

## Requirements

In order to try the *demo*, you'll need to satisfy the following requirements.

### Dependencies

Install PyTorch and torchvision following these [instructions](https://pytorch.org/get-started/locally/).
Then install the remaining dependencies:

    pip install pytorch-lightning bitstring pyldpc

### Dataset

Torchvision should take care by itself about CIFAR10.

### Usage

To test the analyzer-driven workflow using **DenseNet**, CIFAR-10, and **payload.bin**, run:

    ./.venv/bin/python maleficnet_new.py --epochs 10 --model densenet --payload payload.bin --gamma 0.0009 --dataset cifar10 --num_classes 10 --dim 32 --method apoz

The `--method` option selects how model weights are ordered for injection and extraction. Available values are `apoz`, `least_abs`, `least_abs_cluster`, `zscore`, `taylor`, and `combined`. APoZ is used by default when the option is omitted.

The APoZ implementation measures zeros after DenseNet ReLU activations. It computes one score per activation channel, maps that channel to the connected convolutional weights, and ranks channels from the highest APoZ score to the lowest before injection. Weights not covered by a supported ReLU mapping are appended with the least-absolute-value ordering so injector and extractor receive a complete sequence.

For example, use the least-absolute-value baseline with:

    ./.venv/bin/python maleficnet_new.py --epochs 10 --model densenet --payload payload.bin --gamma 0.0009 --method least_abs

### References


### License

`MaleficNet` was made with ♥ and it is released under the MIT license.
