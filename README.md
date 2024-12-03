# NOAAGPROF

NOAAGPROF is a neural-network-based precipitation retrieval for passive-microwave observations from the AMSR2 radiometer. It is based on a prototype for the next-generation GPROF-NN retrievals.

## Overview

This repository provides a Python package called  ``noaagprof``. The ``noaagprof`` package provides site-specific functionality to load training and inference data from NOAA's level 1B input files.

The ``InputLoader`` class is reponsible for loading the retrieval input data from
level-one files and converting it into the format expected by the retrieval.
Additionally, it provides a callback function that is called to convert the raw
retrieval results into the desired format. I hope the code to be sufficiently
well documented to be easy to adapt to NOAA L1 files.


## Installation

### Getting the code

The easiest way to obtain the source code is by cloning the repository using git:

``` shellsession
git clone http://github.com/simonpf/noaagprof
```

### Dependencies

All required dependencies for running GPROF-NN retrievals are collected in the ``noaagprof`` conda environment
defined in the ``noaagprof.yml`` file. Run the following to install and activate it:

``` shellsession
conda env create --file noaagprof.yml
conda activate noaagprof
```

### Installing ``noaagprof``

To use the input loader defined in the ``noaagprof`` package, the package must be install in the current Python environment.
I recomment installing it in editable mode while actively working on it:

``` shellsession
pip install -e /path/to/noaagprof
```

### Downloading the model

The latest noaagprof model is available from [rain.atmos.colostate.edu/gprof_nn/noaagporf/noaagprof.pt](rain.atmos.colostate.edu/gprof_nn/noaagporf/noaagprof.pt). To download it run:

``` shellsession
wget https://rain.atmos.colostate.edu/gprof_nn/noaagporf/noaagprof.pt
```

## Running retrievals

The actual code to run the inference is provided by the ``pytorch_retrieve`` package. ``pytorch_retrieve``
infers all information necessary to run a retrieval from a so-called *inference configuration*. The GPROF-NN
``*.pt`` model files contain such an inference config, which defines the default retrieval behavior. It can be
further customized by providing an explicit path to an inference config file such as the
``noaagprof.toml`` in the repository. The input loader in the inference config file is set  to the ``noaagprof.InputLoader``, which is provided by this package.

### Command line

The retrieval can be run from the command line using the following command:

``` shellsession
pytorch_retrieve inference /path/to/model.pt /path/to/input_files --output_path /path/to/output
```

For additional options to customize inference behavior see ``pytorch_retrieve inference --help``.


### Interactively

Retrievals can also be run interactively using the ``run_inference`` function provided by 
``pytorch_retrieve.inference``. Thils will return a list retrieval results in the form of
``xarray.Datasets`` for all inputs provided by the input loader.

``` python
from pytorch_retrieve.config import InferenceConfig
from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.inference import run_inference
from noaagprof import InputLoader

model = load_model("gprof_nn_3d_amsr2.pt")
loader = InputLoader("/path/to/input_files", config="3d")
# Use default inference settings from model.
inference_config = model.inference_config
results = run_inference(model, loader, inference_config=inference_config)
```




