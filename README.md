# Reproducibility for Exact Transformations of Binary Polynomial Objectives

This repository contains experiment code, model inputs and recorded results for *Exact Transformations of Binary Polynomial Objectives: Auxiliary Variables, Sampling, and Tensor Ranks* by Hadi Salloum, Kirill Novoselov, Aleksandr Pochtarev and Yaroslav Kholodov.

| Experiment | What it checks | Files |
| --- | --- | --- |
| Exact validation | Product-lift penalties, equilibrium Gibbs marginal distortion and tensor unfolding ranks on small instances | [Instructions](experiments/exact_validation/README.md), [protocol](experiments/exact_validation/PROTOCOL.md), [inputs](experiments/exact_validation/inputs), [results](experiments/exact_validation/results) |
| Solver benchmarks | Paired PROTES and TTOpt runs across equivalent formulations, plus finite-precision checks | [Instructions](experiments/solver_benchmarks/README.md), [protocol](experiments/solver_benchmarks/PROTOCOL.md), [models and data](experiments/solver_benchmarks/MODEL_INPUTS.md), [results](experiments/solver_benchmarks/results) |

## Run a small example

Use CPython 3.9 and a separate environment for each experiment. From the directory for that experiment, install the pinned dependencies and use a fresh output path. The smoke commands use only included inputs.

```sh
python -m pip install -r requirements.lock
python run.py smoke --out smoke-output
```

## Rebuild the analysis from recorded results

The `results/` directory in each experiment contains the recorded outputs used in the manuscript. Both analysis commands take those outputs as input and write their reports to a separate directory.

```sh
python report.py --results results --out rebuilt-tables
```

The solver run records include seeds, packed candidate assignments and their losses, evaluation counts and the decoded scores. Their readme describes how to decode them. The independent verifier checks the recorded evaluations without rerunning the optimizers. For the full replay or a fresh experiment, follow the experiment-specific instructions linked above.

## Benchmark data

The small synthetic inputs are included. The two 60-vertex Max-Cut graphs come from the [Biq Mac Library](https://biqmac.aau.at/biqmaclib.html). From `experiments/solver_benchmarks`, use the following commands to retrieve the original graphs, verify their checksums, and independently check the recorded results.

```sh
python fetch_graphs.py --graph-dir graph-cache
python check_independent.py postrun --graphs graph-cache --results results --output replay-check.json
```

When citing the experiment material, use the commit referenced from the manuscript. The pinned dependencies, inputs and recorded results provide the basis for verifying the displayed tables. Fresh solver trajectories and timings can depend on software and hardware.
