"""Execute the prespecified full protocol or the small diagnostic."""
import os
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"qubo-exact-mpl"))
import numpy as np
import matplotlib
from exact_api import API_VERSION, bits, cut_score, cubic_cut, decode_xor, product_lift, check_fiber
from lifts import product_series, gibbs_series
from ranks import degree_checks, rank_series
from report import save, rebuild

HERE = Path(__file__).resolve().parent


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")


def controls():
    cases = json.loads((HERE/"inputs/controls-v1.json").read_text())
    tri = cases["triangle"]
    for z,s in zip(bits(tri["n"]),tri["xor_scores"]):
        assert cubic_cut(tri["edges"],z,tri["u"],tri["v"])==cut_score(tri["edges"],decode_xor(z,tri["u"],tri["v"]))==s
    mix = cases["mixed"]
    for scheme in ("shared","separate"):
        lift = product_lift(mix["n"],mix["terms"],scheme)
        assert [list(g) for g in lift.gates]==mix[scheme+"_gates"]
        if scheme=="shared":
            assert [p for p,n in lift.loads]==mix["P"] and [n for p,n in lift.loads]==mix["N"]
            for p in ("directional","local","global"):
                mags = [abs(c) for s,c in lift.coefficients(p,1).items() if s]
                assert min(mags)==mix[p]["min_abs"] and max(mags)==mix[p]["max_abs"]
        for x in bits(mix["n"]):
            check_fiber(mix["n"],mix["terms"],x,scheme)
    star = cases["star_regression"]
    assert max(cut_score(star["edges"],x) for x in bits(4))==star["cut_max"]
    assert max(i+j+k-c-2*i*j*k for c,i,j,k in bits(4))==star["invalid_surrogate_max"]


def run(mode,out,sha):
    if not __debug__:
        raise RuntimeError("Run without Python optimization")
    smoke = mode=="smoke"
    start = time.perf_counter()
    out.mkdir(exist_ok=True,parents=True)
    inputs = json.loads((HERE/"inputs/instances.json").read_text())
    controls()
    products,fibers,points,circuits = product_series(inputs["polynomials"],smoke)
    gibbs,laws = gibbs_series(smoke)
    ranks,draws = rank_series(inputs,smoke)
    tables = {"products":products,"fibers":fibers,"mixed_points":points,"gibbs":gibbs,
              "probabilities":laws,"ranks":ranks,"degree_checks":degree_checks(smoke)}
    for name,rows in tables.items():
        save(rows,out/(name+".csv"))
    write_json(out/"circuits.json",circuits)
    write_json(out/"coefficient_draws.json",draws)
    rebuild(out,out/"derived")
    info = dict(run_id=f"exact-v1-{mode}",mode=mode,code_sha=sha,api_version=API_VERSION,
                python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
                platform=platform.platform(),machine=platform.machine(),seed_list=inputs["free_draw_seeds"],
                threads={v:os.environ[v] for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS")},
                input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE/"inputs").glob("*.json"))},
                rows={name:len(rows) for name,rows in tables.items()},elapsed_seconds=time.perf_counter()-start)
    write_json(out/"run.json",info)
    print(json.dumps(info,indent=2))


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("full","smoke"))
    p.add_argument("--out",type=Path)
    p.add_argument("--code-sha",help="Override the automatically resolved code revision")
    a = p.parse_args()
    sha = a.code_sha or subprocess.check_output(["git","log","-1","--format=%H","--","*.py","inputs","PROTOCOL.md","requirements.lock","README.md"],cwd=HERE,text=True).strip()
    run(a.mode,a.out or Path("pilot" if a.mode=="smoke" else "results"),sha)
