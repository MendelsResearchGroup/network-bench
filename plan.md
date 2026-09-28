Read this plan. Before acting, ask ANY questions if something is unclear. Do not assume any defaults before confirming via explicit question. Present a step by step plan. After that, if given permission, execute it step by step. After each step, pause and make sure everything works as expected/desired.



# A general benchmark for dataset and model comparison

Our data are compression simulation trajectories generated using LAMMPS MD package. Each trajectory is a list of pytorch geometric Data objects (see datasets directory). 
Learning on that, we have torch based autoregressive simulator models that predict the evolution of the system.

We are interested in a repo where we can simply compare the performance of different models on the same sets of data. Meaning that a person in the lab, can come up with a new ML/torch-based model and plug it in to see how well it fares against existing models. 

A dataset will have a unique key defined by the user, and also a model will have. Additionally, models will define hyperparameters that can be changes like hidden_size and others. Then we want to cache these reuslts (for later eval) based on these keys, including a list of changing hyperparameters between the models. These should be in a simple dictionary/list/json somewhere to define.

Essentially the project has 2 types of interfaces, the data structure, where I want you to read the existing data structure and generate a minimal python interface of the Data object, such that users will know what is is stored in the edge_attr and such.  Then we also want to have an interface for a model to consume, it gets as input some data frames and outputs the predicted such that a user 


1. The input graph for each model is constructed globally within the benchmark. We construct input graph with different node and edge features. An important metric is history: the number of previous frames used to construct the input graph. Additonally, several different node features can be computed and used in any combination: global/fractional particle coordinates, total/residual particle velocites, per-particle net forces, etc. On other other hand, edge features are typically used as provided in the dataset. A Model with a proper intefrace should be able to take the provided input graph.

2. The output of the model should happen within the model itself and is not the job of this benchmark. A model should output the next state in the trajecotry, with `graph` identical in structure to the raw dataset Data objects.

3. The whole training and evaluation of a model happens inside this project, i.e., the loss function is provided by the benchmark.


## Dataset Difficulty metrics 

Before getting to models, we first can evaluate the datasets difficulty by a few metrics. Take the metric from the provided data_comparison.ipynb.

## Model perfromance metrics

Each model is evaluated based on its ability to accurately predict a trajectory. We are interested, among others, in the following: last rollout step position MSE, relative MSE (MSE / frozen MSE), $R^2$ of the predicted and GT Poisson's ratios, $R^2$ of the predicted and GT stress.

## References

The reference can be found in /home/work/KG_chains. It contains all the necessary models and way to train and evalulate but we want you to extract miniminal needed pieces into this standalone project. Reuse existing code as much as possible

## Technical guidelines

This is a standalone project, but I want the data and model interface somehow exported such that people can import it from the outside by an import from Github.

This project will live in an cpu HPC cluster 

Should be a straighforward and humanely readable project, don't add overhead for checking different inputs in places. It's fine to check that the consumed data or model fits the interface but no need at other places unless is critical.
 
No need to any kind of tests in project currently

Ask before doing anything!!!!!


