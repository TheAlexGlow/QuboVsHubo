"""CPU solver calls and time measurement shared by the pilot and main runs."""
import os
import tempfile
from pathlib import Path
from time import perf_counter

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "false"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"qubo-b3-mpl"))
import numpy as np
from protes import protes
from ttopt import TTOpt

SETTINGS = {
    "PROTES": dict(k=64, k_top=8, k_gd=1, lr=0.05, r=3),
    "TTOpt": dict(rank=4, fs_opt=1.0, add_opt_inner=True, add_opt_outer=False,
                  add_opt_rect=False, add_rnd_inner=False, add_rnd_outer=False),
}


def solve(engine, dimension, seed, budget, oracle):
    points, losses = [], []

    def get_loss(batch):
        batch = np.asarray(batch)
        if batch.ndim != 2 or batch.shape[1] != dimension or not np.all((batch==0)|(batch==1)):
            raise ValueError("Invalid candidate batch")
        data = batch.astype(np.uint8, copy=True)
        # Give the objective signed integers to avoid subtraction in uint8.
        values = np.asarray(oracle(data.astype(np.int64)))
        if values.shape != (len(batch),) or not np.all(values == values.astype(np.int32)):
            raise ValueError("Expected one exact signed int32 energy per point")
        points.append(data)
        losses.append(values.astype(np.int32))
        return values.astype(float)

    proposed_batches = []

    class CountedTTOpt(TTOpt):
        def comp_opt(self, I, *args):
            if self.k_evals < self.evals:
                proposed_batches.append(len(I))
            return super().comp_opt(I, *args)

    start = perf_counter()
    config = SETTINGS[engine]
    if engine == "PROTES":
        if budget % config["k"]:
            raise ValueError("PROTES budgets must be multiples of the sampling batch size")
        returned, value = protes(get_loss, dimension, 2, m=budget, seed=seed, info={},
                                 is_max=False, **config)
    else:
        optimizer = CountedTTOpt(get_loss, d=dimension, n=2, evals=budget, is_func=False,
                          is_vect=True, with_cache=False)
        optimizer.optimize(seed=seed, is_max=False, **config)
        returned, value = optimizer.i_opt, optimizer.y_opt
    returned, value = np.asarray(returned, dtype=np.uint8), float(value)
    wall = perf_counter() - start
    actual = [len(p) for p in points]
    data, energies = np.concatenate(points), np.concatenate(losses)
    if sum(actual) != budget or value != min(energies):
        raise RuntimeError("Solver budget or returned loss differs from the recorded evaluations")
    where = np.flatnonzero(np.all(data == returned, axis=1))
    if not len(where) or energies[where[0]] != value:
        raise RuntimeError("Returned candidate absent from the evaluation log")
    return data, energies, actual, int(where[0]), wall, (proposed_batches or actual)


if __name__ == "__main__":
    import json
    import sys
    # A held-out, 60-vertex unit-weight cycle is used only for timing. It is
    # excluded from all benchmark tables and does not select representations.
    def timing_loss(x):
        return -np.count_nonzero(x != np.roll(x,1,axis=1),axis=1)
    out = {}
    for solver in SETTINGS:
        _, _, batches, _, seconds, proposed = solve(solver, 60, 17, 4096, timing_loss)
        out[solver] = dict(seconds=seconds, n_evals=sum(batches), n_batches=len(batches),
                           last_batch=batches[-1], proposed_last_batch=proposed[-1])
    Path(sys.argv[1]).write_text(json.dumps(dict(profile="pilot",case="cycle60",seed=17,
                                       budget=4096,results=out),indent=2)+"\n")
