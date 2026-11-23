"""Run the seeded paired design on one CPU process and archive every queried point."""
import argparse
import base64
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import tempfile
import zlib

from engines import np, solve, SETTINGS
from models import CASE_IDS, formulations, load

HERE = Path(__file__).resolve().parent
PROFILES = {
    "smoke": (("triangle","mixed"), (200,), 128),
    "full": (CASE_IDS, tuple(range(201,209)), 4096),
}


def checkpoint(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False)+"\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(encoded)
    os.replace(tmp, path)


def packed(data):
    return base64.b64encode(zlib.compress(data.tobytes(order="C"))).decode("ascii")


def code_revision():
    # Follow code/input history only. The result commit is its direct descendant.
    tracked = [p.name for p in HERE.glob("*.py")] + ["inputs","requirements.lock","PROTOCOL.md","README.md","MODEL_INPUTS.md"]
    return subprocess.check_output(["git","log","-1","--format=%H","--",*tracked],cwd=HERE,text=True).strip()


def entry(model, form, points, losses, idx):
    bits = points[idx].astype(int)
    x, score, feasible = model.decode(form,bits)
    return dict(bits=bits.tolist(), decoded=np.asarray(x).tolist(), score=int(score),
                feasible=bool(feasible),loss=int(losses[idx]),eval_index=int(idx)+1)


def perform(model, meta):
    engine, form = meta["solver"], meta["form"]
    points, losses, batches, index, seconds, proposed = solve(
        engine, model.dimension(form), meta["seed"], meta["budget"], lambda batch:model.evaluate(form,batch))
    if model.maximize:
        scores = -losses
        running = np.maximum.accumulate(scores)
    else:
        scores = model.evaluate("original",points[:,:model.n])
        running = np.minimum.accumulate(scores)
    improvements = np.flatnonzero(np.r_[True, running[1:] != running[:-1]])
    returned = entry(model,form,points,losses,index)
    best = entry(model,form,points,losses,improvements[-1])
    if best["score"] != running[-1] or returned["score"] != scores[index]:
        raise ValueError("Decoded score disagrees with trajectory")
    return dict(**meta,exit_code=0,termination="evaluation_budget",dimension=points.shape[1],
                n_evals=len(points),seconds=seconds,batches=batches,proposed_batches=proposed,
                n_distinct=len(np.unique(points,axis=0)),optimum=model.optimum,maximize=model.maximize,
                returned=returned,best_original=best,trace=[[int(i)+1,int(scores[i])] for i in improvements],
                eval_bits_b64=packed(np.packbits(points,axis=1,bitorder="big")),
                eval_loss_b64=packed(losses.astype("<i4")))


@contextmanager
def timeout(seconds):
    def expired(*unused):
        raise TimeoutError(f"Run exceeded {seconds} seconds")
    old = signal.signal(signal.SIGALRM,expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,old)


def run(args):
    case_ids,seeds,budget = PROFILES[args.profile]
    output = args.out.resolve()
    runs = output/"runs"
    runs.mkdir(parents=True,exist_ok=True)
    # The OS releases this lock if a process dies partway through a local run.
    lock = (output/".runner.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sha = code_revision()
    models = {k:load(k,args.graph_dir) for k in case_ids}
    jobs = []
    for case in case_ids:
        for form in formulations(case):
            for solver in SETTINGS:
                for seed in seeds:
                    run_id = f"{args.profile}-{case}-{solver}-{form}-{seed}"
                    jobs.append(dict(run_id=run_id,case=case,instance_id=case,form=form,solver=solver,
                                     seed=seed,budget=budget,config=SETTINGS[solver],code_sha=sha))
    random.Random(11023).shuffle(jobs)
    checkpoint(output/"design.json",dict(profile=args.profile,code_sha=sha,budget=budget,seeds=list(seeds),
               platform=platform.platform(),python=platform.python_version(),run_order=[j["run_id"] for j in jobs]))
    for i,job in enumerate(jobs,1):
        file = runs/(job["run_id"]+".json")
        if file.exists():
            old = json.loads(file.read_text())
            if any(old.get(k) != v for k,v in job.items()):
                raise ValueError(f"Existing {file.name} has a different configuration; choose a fresh output directory")
            if old["exit_code"] == 0 or not args.retry_errors:
                continue
            attempts = output/"attempts"
            attempts.mkdir(exist_ok=True)
            attempt = len(list(attempts.glob(file.stem+"-*.json"))) + 1
            checkpoint(attempts/(file.stem+f"-{attempt}.json"),old)
        try:
            with timeout(120):
                result = perform(models[job["case"]],job)
        except Exception as error:
            result = dict(**job,exit_code=124 if isinstance(error,TimeoutError) else 1,
                          n_evals=None,termination=type(error).__name__,message=str(error))
        checkpoint(file,result)
        if i%16==0 or i==len(jobs) or result["exit_code"]:
            print(i,len(jobs),job["run_id"],result["exit_code"],flush=True)
    records = [json.loads((runs/(job["run_id"]+".json")).read_text()) for job in jobs]
    coverage = dict(planned=len(jobs),completed=sum(r["exit_code"]==0 for r in records),
                    errors=sum(r["exit_code"] not in (0,124) for r in records),
                    timeouts=sum(r["exit_code"]==124 for r in records),missing=0,
                    billed_successful_points=sum(r["n_evals"] for r in records if r["exit_code"]==0))
    checkpoint(output/"coverage.json",coverage)
    print(json.dumps(coverage))
    lock.close()
    return int(coverage["completed"] != len(jobs))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile",choices=PROFILES)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--graph-dir",type=Path)
    p.add_argument("--retry-errors",action="store_true")
    sys.exit(run(p.parse_args()))
