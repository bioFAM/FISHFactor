# FISHFactor: A Probabilistic Factor Model for Spatial Transcriptomics Data with Subcellular Resolution
Code repository supplementing the [eponymous paper](https://www.biorxiv.org/content/10.1101/2021.11.04.467354v1)

FISHFactor is a non-negative, spatially informed factor analysis model with a Poisson point process likelihood to model single-molecule resolved data, as for example obtained from multiplexed fluorescence in-situ hybridization methods. In addition, FISHFactor allows to integrate multiple data groups (e.g. cells) by jointly inferring group-specific factors and a weight matrix that is shared across groups. The model is implemented with the deep probabilistic programming language [Pyro](https://pyro.ai/) and the Gaussian process package [GPyTorch](https://gpytorch.ai/).

<img src="model.png" width=80% height=80%>


## Repository structure
- **src/** contains the FISHFactor model, data simulation and util functions.
- **exp_sim/** contains experiments on simulated data (section 4.1 in paper).
- **exp_subsample/** contains data subsampling experiments on NIH/3T3 cells from the seqFISH+ paper (section 4.2 in the paper).
- **exp_multicell/** contains experiments with multiple NIH/3T3 cells (section 4.2 in the paper).
- **data/** contains scripts to download and process seqFISH+ data as well as scripts to simulate data.

## Usage

An example for using FISHFactor with 15 NIH/3T3 cells is shown in *example.ipynb*.
