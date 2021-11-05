# FISHFactor: A Probabilistic Factor Model for Spatial Transcriptomics Data with Subcellular Resolution
Code repository supplementing the eponymous paper.

FISHFactor is a non-negative, spatially informed factor analysis model with a Poisson point process likelihood to model single-molecule resolved data, as for example obtained from multiplexed fluorescence in-situ hybridization methods. In addition, FISHFactor allows to integrate multiple data groups (e.g. cells) by jointly inferring group-specific factors and a weight matrix that is shared across groups. The model is implemented with the deep probabilistic programming language [Pyro](https://pyro.ai/) and the Gaussian process package [GPyTorch](https://gpytorch.ai/).

<img src="model.png" width=80% height=80%>


## Repository structure
- **src/** contains the FISHFactor model, data simulation and util functions.
- **exp_sim/** (*in preparation*) contains experiments on simulated data (section 4.1 in paper).
- **exp_subsample/** (*in preparation*) contains data subsampling experiments on NIH/3T3 cells from the seqFISH+ paper (section 4.2 in the paper).
- **exp_multicell/** (*in preparation*) contains experiments with multiple NIH/3T3 cells (section 4.2 in the paper).
- **exp_scalability/** (*in preparation*) contains scalability experiments on NIH/3T3 cells (section 4.2 in the paper).
- **data/** (*in preparation*) contains scripts to download and process seqFISH+ data as well as scripts to simulate data.

## Usage

*in preparation*