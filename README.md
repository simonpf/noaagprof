# NOAAGPROF

This repository contains documentation and required code to run GPROF-NN retrievals for AMSR2.

## Overview

This repository contains a Python package called  ``noaagprof`` providing a single class ``InputLoader``.
The ``InputLoader`` is reponsible for loading the retrieval input data from level-one files
and converting it into the format expected by the retrieval. Additionally, it provides a callback
function that is called to convert the raw retrieval results into the desired format. I hope the code
to be sufficiently well documented to be easy to adapt to NOAA L1 files.

## Installation

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

## Running retrievals

The actual code to run the inference is provided by the ``pytorch_retrieve`` package. ``pytorch_retrieve``
infers all information necessary to run a retrieval from a so-called *inference configuration*. The GPROF-NN
``*.pt`` model files contain such an inference config, which defines the default retrieval behavior. It can be
further customized by providing an explicit path to an inference config file such as the
``inference_gprof_nn_3d.toml`` in the repository. I have already set the input loader in the inference config to
the ``noaagprof.InputLoader``, so by default those models will use the NOAAGPROF input loader for the
retrievals.

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
from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.inference import run_inference
from noaagprof import InputLoader

model = load_model("gprof_nn_3d_amsr2.pt")
loader = InputLoader("/path/to/input_files", config="3d")
# Use default inference settings from model.
inference_config = model.inference_config
results = run_inference(model, loader, inference_config=model.inference_config)
```




