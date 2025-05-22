
# Installation

Our code builds on [hydra](https://hydra.cc) to run experiments and compose experiment configurations. To install all dependencies, run

```bash
pip install -e .
```
This will install our code as a package `graph_al`. 

# Running our Code

You can now run an Active Learning (AL) run via config file(s) using:

```bash
python main.py model=gcn data=cora_ml acquisition_strategy=aleatoric_propagated data.num_splits=5 model.num_inits=5 print_summary=True
```

This will run an AL run on the `CoraML` dataset using a `GCN` backbone, the aleatoric uncertainty as acquisition function starting with one random sample per class and acquiring until a budget of `5 * num_classes` is exhausted.

## Weights & Biases

Optionally, we support W&B for logging purposes. By default, it is disabled. You can enable it by passing: `wandb.disable=False`. You can configure further W&B options using the `wandb` subconfiguration (see `graph_al/config.py` -> `WandbConfig`, default values in `config/main.yaml`)

# Configuring an Experiment

All configurations are found in `config`. They are loaded and composed using [hydra](https://hydra.cc). The base default configuration is `config/main.yaml`. 

### Output
The `output_base_dir` field determines where outputs are logged. Note that the default value uses the `wandb.group` and `wandb.name` fields, even if W&B is disabled. This eases the tracking of experiments. You do not have to do this, you can pass any value to `output_base_dir`, e.g. `output_base_dir=my/output/dir`.

It will create a directory for the run in `output_base_dir` where masks (indices of the acquired nodes), model checkpoints, per-run metrics and aggregate metrics are stored. The latter (`acquisition_curve_metrics.pt`) contains the aggregate results over all splits and initializations. It serializes a dict that maps metrics to (nested) lists of values. You can also pass `print_summary=True` to print results.
Note that metrics like the AUC you need compute yourself from the `acquisition_curve_metrics.pt`.

### Data

You can find default configurations for the datasets in `config/data` and load them with e.g. `data=cora_ml`. 

**CSBM**: The CSBMs used in our paper can be generated using the config `csbm_100_7` (CSBM with 100 nodes and 7 classes) or `csbm_1000_4` (CSBM with 1000 nodes and 4 classes). The cache precomputed likelihoods in `${output_base_dir}/likelihood_cache`.

### Models

You can find different model configurations in `config/models`. For datasets with an explicit generative process, use the `bayes_optimal` configuration.


### Acquisition Strategy

Configurations for acquisition strategies are found in `config/acquisition_strategy`. For the initial pool, different "initial acquisition strategies" can be used: In our paper, we always use `config/initial_acquisition_strategy/balanced.yaml`, which will yield a pool with a balanced class distribution, i.e. 1 label per class.


# Code Structure

We devise a code structure that is easily extendable for new datasets, models and acquisition strategies for AL.

The code is structured into four main modules:
- `graph_al/data`: The main module for graph datasets. The `Data` class in `graph_al/data/base.py` extends a torch-geometric `Data` instance with utility functions while the `BaseDataset` wraps the data (and encapsulates meta information about its generative process). A base class for explicit generative processes is provided in `graph_al/data/generative.py` and CSBMs are implemented in `graph_al/data_sbm`.
- `graph_al/model`: Provides different GNN models. Models should inher `graph_al/model/base.py`'s `BaseModel` class such that they can be ensembled and fit into the pipeline of our code.
- `graph_al/trainer`: Training code for different GNN models.
- `graph_al/acquisition_strategy`: Different acquisition strategies for AL. All strategies inherit from `graph_al/acquisition_strategy/base.py`'s `BaseAcquisitionStrategy`. Most strategies assign a quantity to each node and pick the node with the highest / lowest quantity. Such strategies are best implemented by subclassing `graph_al/acquisition_strategy/prediction_attribute.py`'s `AcquisitionStrategyByPredictionAttribute` by implementing a `get_attribute` method that assigns each node a quantity.

Adding a new dataset, model or strategy boils down to:
1. Sublcassing the corresponding base module
2. Writing a configuration (i.e. a dataclass) for the module (see `build.py` of the corresponding module) and specify default values
3. Registering the default values for this configuration in hydra (see `config.py` of the corresponding module, where `cs.store` is called).
4. Writing a yaml file in `config/...` with further default values so that you can access your module from the CLI


