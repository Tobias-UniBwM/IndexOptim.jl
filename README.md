# IndexOptim

This Julia package provides the software code for optimizing spectral indices. It accompanies the publication [_Optimized Spectral Indices for Camouflage Detection in Multispectral Imagery_](https://www.tandfonline.com/journals/tgrs20), which contains detailed background information and a comprehensive experimental evaluation of this software tool in the context of camouflage detection. In addition, the package supports arbitrary index structures, custom optimizers, datasets and fitness descriptors.

## Structure

### Source Files ([src](src))

- [dataset.jl](src/dataset.jl): contains everything related to data loading, manipulation and filtering
- [index.jl](src/index.jl): contains tools to construct optimizable index definitions and fitness descriptors
- [optim.jl](src/optim.jl): contains optimizers
- [config.jl](src/config.jl): provides a config-file-based command line interface for the package
- [tvi.jl](src/tvi.jl): Target Visibility Index (TVI) calculation, the optimization criterion in the accompanying study

### Config Files ([config](config))

- [train.yaml](config/train.yaml): generic config template for optimizing an arbitrary index
- [eval.yaml](config/eval.yaml): generic config template for testing an arbitrary optimized index

### Experiments ([experiments](experiments))

Contains the configuration files used in the experiments of the publication accompanying this package. [MUDCAD-X](https://github.com/Tobias-UniBwM/MUDCAD-X) is required to execute them. See the `captures` property in the dataset definition of the config files to define a custom path to the dataset.

- [lri6](experiments/lri6/): configs for training and testing the $LRI_6$
- [lri2](experiments/lri2/): configs for training and testing the $LRI_2$
- [lrind](experiments/lrind/): configs for training and testing the $LRI_{nd}$
- [lrir](experiments/lrir/): configs for training and testing the $LRI_r$

## Usage

### Installation

Clone the repository and set up the corresponding Julia environment:.

```sh
git clone https://github.com/Tobias-UniBwM/IndexOptim.jl.git
cd IndexOptim.jl
julia --project -e "using Pkg; Pkg.instantiate()"
```

### General

The package is primarily intended to be used with config files. The [`run.jl`](run.jl) file provides easy access to the command line interface. In order to perform an index optimization, execute the following statement.
```sh
julia --project run.jl path/to/your/config.yaml
```
### Experiments

The [experiments](experiments/) directory contains the configurations to run the optimizations presented in the accompanying publication. All require [MUDCAD-X](https://github.com/Tobias-UniBwM/MUDCAD-X) to be available right in the packages' root directory. Therefore, the dataset needs to be downloaded first:
```sh
git clone https://github.com/Tobias-UniBwM/MUDCAD-X.git
```

#### Optimization

The configuration files for index optimization are named `config_train.yaml`. For example, to optimize the Linear Ratio Index utilizing all six single-channel bands of MUDCAD-X ($LRI_6$), execute the following command:
```sh
julia --project run.jl experiments/lri6/config_train.yaml
```
This starts the optimization procedure and logs the progess to stdout and a tensorboard events file. The events file will be located in the [results](experiments/lri6/results/train/) directory. Visit [http://localhost:6006/](http://localhost:6006/) to monitor progress after executing the following command:
```sh
tensorboard --logdir experiments/lri6/results/train/tensorboard
```
Note that tensorboard might not be available on your system. It can be installed using pipx by executing the following command:
```sh
pipx install tensorboard
```

#### Evaluation

The configuration files for index optimization are named `config_test.yaml`. In order to evaluate the $LRI_6$ on the test dataset split from MUDCAD-X, execute the following command:
```sh
julia --project run.jl experiments/lri6/config_test.yaml -t eval
```
Note that this provides only an average value of the TVI over the entire test dataset and all target classes. Average TVIs for specific target classes can be obtained by defining a category filter in the dataset definition of the config file. For example, to determine the average TVI for the `net2dgreen` target class include the following category filter after the captures and channels filter:
```yaml
- filter:
    # captures:
    # ...
    # channels:
    # ...
    categories:
      names: ["net2dgreen"]
      invert: false
```

## Extension

This package can be easily extended by new datasets, optimizers, index transforms, index reductions and fitness descriptors. For optimizers, see the `AbstractIndexOptimizer` interface, which is implemented by multiple data types in [optim.jl](src/optim.jl). Fitness descriptors are added by implementing the `AbstractIndexFitnessDescriptor` interface defined in [index.jl](src/index.jl). There are also the interfaces required for new optimizable index structures, transforms and reductions. Datasets can be added by implementing the `AbstractDataset` interface given in [dataset.jl](src/dataset.jl).

In order to include new functionality in the config-file-based command line interface, an appropriate parser function must be defined and registered in the respective config parser map defined in [config.jl](src/config.jl). For example, `register_optimizer_config_parser` must be called in a modules' `__init__` function to register a new optimizer:

```julia
module MyIndexOptimizationModule

using IndexOptim

# code...

function __init__()
    register_optimizer_config_parser("incredible_optimizer", process_incredible_optimizer_config)
end

end
```

Under the optimizer section in the config file, the additional option `incredible_optimizer` is then supported:
```yaml
# config...
optimizer:
  # config...
  incredible_optimizer:
    # config...
```

## Cite

If you use this package in your research please cite the following publication:
```
@Article{Hupel2025,
  author    = {Hupel, Tobias and St{\"{u}}tz, Peter},
  journal   = {GIScience \& Remote Sensing},
  title     = {Optimized Spectral Indices for Camouflage Detection in Multispectral Imagery},
  year      = {2025},
  issn      = {1548-1603},
  doi       = {10.1080/15481603.2025.2508574},
  publisher = {Taylor \& Francis},
}
```