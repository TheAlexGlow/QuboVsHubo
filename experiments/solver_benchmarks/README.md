# Reproduce the solver benchmarks

Run the following commands from experiments/solver_benchmarks in a checkout of the solver branch. The recorded environment used CPython 3.9.6 on arm64 macOS. Use a fresh environment and install requirements.lock with pip. The pure-Python objective functions, case definitions and control data are included in this directory. Graphs are downloaded from the Biq Mac source and verified by SHA-256.

~~~sh
python run.py smoke --out /tmp/qubo-solver-smoke
python reproduce.py full --out /tmp/qubo-solver-full
python report.py --results results --out /tmp/qubo-solver-tables
~~~

The full entry point performs the independent gate checks, resolves the verified graph cache, runs or resumes the complete design, rebuilds reports, verifies candidate and trajectory records, and executes the quantization study. Use QUBO_GRAPH_DIR or --graph-dir to select a cache containing the original Biq Mac files. MODEL_INPUTS.md gives independent data acquisition and bit ordering. To explicitly retry recorded nonzero exits use reproduce.py full --retry-errors. The small smoke command requires no graph downloads. Run the following for independent checks of the committed results.

~~~sh
python check_independent.py preflight --results /tmp/qubo-preflight
python check_independent.py postrun --results results
~~~

## Raw and derived records

design.json lists the scheduled run IDs in execution order, host platform, Python version, budget, seeds and the commit containing the code. coverage.json counts completions, missing runs, errors, timeouts and billed points. In runs/ each JSON identifies case, formulation, solver, seed, configuration and termination. Batch arrays count actual and originally proposed points. n_distinct counts distinct encoded points. Both returned (native optimum) and best_original (best decoded candidate seen) contain the encoded and decoded bits, natural score, loss, feasibility and one-based evaluation index. For maximization, loss is negative cut weight. For minimization with auxiliaries, it is the QUBO energy. The trace lists the first point and all strict improvements in the natural objective.

Every query is included. Base64-decode and zlib-decompress eval_bits_b64, reshape the bytes to n_evals by ceil(dimension/8), unpack each row using big-endian bit order and trim padding to dimension. Decode eval_loss_b64 with the same two outer steps and read little-endian signed 32-bit integers in row order. Loss evaluation is deterministic given the inputs. Raw score records can be reconstructed on a different platform without repeating stochastic solver runs. Hardware, compiled backends and package versions can affect runtime and solver trajectories.

The independent audit compares per-form medians with derived/by_form.csv and replays the packed streams. The remaining CSVs give seed-level contrasts, contrast summaries, trajectory quantiles and run inclusion. The PDF and SVG figures derive from those same runs. precision.csv records exact rational fiber errors for each separately tested quantization. See PROTOCOL.md for inference scope, timing, signs and the declared limits.
