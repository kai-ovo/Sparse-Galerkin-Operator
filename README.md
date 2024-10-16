# Sparse Galerkin Operator
Codes for the paper "Reducing Operator Complexity of Galerkin Coarse-Grid Operators with Machine Learning". SIAM Journal on Scientific Computing (2024): S296-S316.

# Details

- PDE classes are under the directory `libs`
- Training scripts are provided in `notebooks`
- Spectrum results are provided in `spectrum`
- Codes to reproduce the baseline results are in `SparseAggregation`. These codes are debugged from the original paper's codes released in 2016 by myself to fix compatibility issue and MEX code bugs. They can only be run on MATLAB online, as MATLAB has removed OpenMP support on their major software. An example script for reproducing the baseline results are given in `test_baseline.m`. Other data can be constructed using scripts in `notebooks` or obtained by contacting the author.
- Also included a preliminary work on sparsifying Galerkin Operators on unstructured meshes using Graph Attention Transformer (GAT). This algorithm is **NOT** in the paper
