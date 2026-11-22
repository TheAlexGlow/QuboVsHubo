"""Independent truth-table, probability and exact-rank audits of the saved results."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from fractions import Fraction
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def bit_string(value):
    require(value and set(value) <= {"0", "1"}, ("non-binary", value))
    return tuple(map(int, value))


def gate_cost(a, b, u, v, w):
    # Direct table by the number of true parents, independent of polynomial collection.
    return ((0, a + 2*b), (0, b), (a, 0))[u + v][w]


def check_originals():
    # Literal scalar transcriptions of (46)-(49), read in submission order.
    originals = {
        "f1": lambda a,b,c,d,e,f: -3*a*c*f+6*b*a*c-c*d*e*f+c*f+2*b*e-e-a,
        "f2": lambda a,b,c,d: 5*a*b*c-4*b*c*d-2*a*c*d+a*c+b-d,
        "f3": lambda a,b,c,d,e,f: -2*a*b*c*d*e-a*c*d*e*f+2*a*b*c*f+b*c*d*e-3*a*e+b*e,
        "f4": lambda a,b,c,d,e,f,g: (-2*a*b*c*d*e+b*c*e*f*g-3*b*c*d*e+a*b*c*d
                                      -c*e*f*g+2*a*b*c-2*b*c*f+a*f-f*g-g),
    }
    source = json.loads((HERE/"inputs"/"instances.json").read_text())["polynomials"]
    stats = {}
    for model, oracle in originals.items():
        spec = source[model]
        values = []
        for x in product(range(2), repeat=spec["n"]):
            f = oracle(*x)
            readout = sum(c*math.prod(x[i] for i in monomial) for c,monomial in spec["terms"])
            require(f == readout, (model,x,f,readout))
            values.append(f)
        stats[model] = {"points": len(values), "minimum":min(values), "argmins":values.count(min(values))}
    return stats


def check_mixed(rows):
    keys = set()
    settings = set()
    for r in rows:
        require(r["model"] == "mixed", r)
        x = bit_string(r["x"])
        y = bit_string(r["y"])
        d = Fraction(r["delta"])
        scheme, penalty = r["scheme"], r["penalty"]
        keyspec = (scheme, penalty, d)
        settings.add(keyspec)
        key = (*keyspec, x, y)
        require(key not in keys, ("duplicate", key))
        keys.add(key)

        shared = scheme == "shared"
        require(shared or scheme == "separate", scheme)
        require(len(x) == 5 and len(y) == (2 if shared else 4), key)
        f = math.prod(x[:3]) * (2*x[3] - 3*x[4])
        q = 2*y[1]*x[3] - 3*y[-1]*x[4]
        bad = 0
        for start in range(0, len(y), 2):
            p, n = ((2, 3) if shared else ((2, 0) if start == 0 else (0, 3)))
            a, b = {"directional": (p+d, n+d), "local": (max(p,n)+d,)*2,
                    "global": (5+d,)*2}[penalty]
            w, t = y[start:start+2]
            q += gate_cost(a, b, x[0], x[1], w) + gate_cost(a, b, w, x[2], t)
            bad += (w != x[0]*x[1]) + (t != w*x[2])
        require((Fraction(r["f"]), Fraction(r["q"]), int(r["violations"])) == (f, q, bad), key)
        require(q - f >= d * bad, key)
    expected_settings = {(scheme, policy, 1) for scheme in ("shared","separate")
                         for policy in ("directional","local","global")}
    expected_settings.add(("shared","directional",0))
    require(settings == expected_settings, ("mixed configurations",settings))
    require(len(keys) == sum(2**(7 if s == "shared" else 9) for s, _, _ in settings), "mixed coverage")
    return len(keys)


def check_gibbs(rows):
    # Integers and rational probabilities at beta=log(2) avoid log-sum-exp entirely.
    selected = [r for r in rows if r["family"] == "sharpness" and int(r["m"]) == 2
                and float(r["delta"]) == 1 and abs(float(r["beta"]) - math.log(2)) < 1e-12]
    values = {}
    for r in selected:
        mpen = int(float(r["M"]))
        require(mpen not in values, ("repeated sharpness", mpen))
        fibers = []
        for x in range(2):
            weight = Fraction(3, 2) if x else Fraction(1)
            fibers.append(weight * sum(Fraction(1, 2)**((1+x*mpen)*(i+j))
                                       for i,j in product(range(2), repeat=2)))
        exact = abs(fibers[0]/sum(fibers) - Fraction(2, 5))
        require(abs(float(r["tv"]) - float(exact)) < 1e-12, r)
        require(abs(float(r["b1"]) - 0.2) < 1e-12 and abs(float(r["b2"]) - 0.25) < 1e-12, r)
        values[mpen] = str(exact)
    require({0,1,2} <= values.keys(), "sharpness grid misses rational controls")
    # Independently sum both laws on the product lift in exact dyadic arithmetic.
    mass, joint = [], []
    for x in product(range(2), repeat=5):
        f = math.prod(x[:3]) * (2*x[3]-3*x[4])
        mass.append(Fraction(2)**(-f))
        fiber = Fraction(0)
        for w, t in product(range(2), repeat=2):
            q = t*(2*x[3]-3*x[4]) + gate_cost(3,4,x[0],x[1],w) + gate_cost(3,4,w,x[2],t)
            fiber += Fraction(2)**(-q)
        joint.append(fiber)
    tv = sum(abs(u/sum(mass)-v/sum(joint)) for u,v in zip(mass,joint))/2
    ordinary = [r for r in rows if r["family"] != "sharpness" and int(r["m"]) == 2
                and float(r["delta"]) == 1 and abs(float(r["beta"])-math.log(2)) < 1e-12]
    require(len(ordinary) == 1 and abs(float(ordinary[0]["tv"])-float(tv)) < 1e-12, ordinary)
    return {"sharpness": values, "ordinary_m2": str(tv)}


def real_rank(matrix):
    a = [[Fraction(x) for x in row] for row in matrix]
    pivot = 0
    for j in range(len(a[0])):
        row = next((i for i in range(pivot, len(a)) if a[i][j]), None)
        if row is None:
            continue
        a[pivot], a[row] = a[row], a[pivot]
        for i in range(pivot+1, len(a)):
            if a[i][j]:
                factor = a[i][j] / a[pivot][j]
                a[i] = [x - factor*y for x, y in zip(a[i], a[pivot])]
        pivot += 1
    return pivot


def copy_rank(order, k, m=2):
    labels = list(range(2*m)) if order == "grouped" else [j for i in range(m) for j in (i, i+m)]
    values = []
    for z in product(range(2), repeat=2*m):
        bits = dict(zip(labels, z))
        mismatch = sum(bits[i] ^ bits[i+m] for i in range(m))
        values.append(Fraction(1, 2)**mismatch)
    width = 2**(2*m-k)
    return real_rank([values[i:i+width] for i in range(0, len(values), width)])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_rebuild(results):
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        subprocess.run([sys.executable, str(HERE / "report.py"), "--results", str(results),
                        "--out", str(fresh)], check=True)
        rebuilt = {p.relative_to(fresh).as_posix(): digest(p) for p in fresh.rglob("*") if p.is_file()}
    published = results / "derived"
    recorded = {p.relative_to(published).as_posix(): digest(p) for p in published.rglob("*") if p.is_file()}
    require(rebuilt and rebuilt == recorded, ("derived table/figure mismatch", sorted(rebuilt), sorted(recorded)))
    return rebuilt


def graph_rank(spec, order, coordinate_map, cut, gibbs):
    n = spec["n"]
    values = []
    for assignment in product(range(2), repeat=n):
        z = tuple(assignment[order.index(i)] for i in range(n))
        x = list(z)
        if coordinate_map == "xor":
            x[spec["u"]] ^= z[spec["v"]]
        elif coordinate_map == "tree_edge":
            for i in range(n):
                parent = spec["parents"][i]
                while parent >= 0:
                    x[i] ^= z[parent]
                    parent = spec["parents"][parent]
        else:
            require(coordinate_map == "identity", coordinate_map)
        score = sum(w for i,j,w in spec["edges"] if x[i] != x[j])
        values.append(Fraction(2)**score if gibbs else score)
    width = 2**(n-cut)
    return real_rank([values[i:i+width] for i in range(0, 2**n, width)])


def check_ranks(rows):
    copy, graph_rows = {}, {}
    graphs = json.loads((HERE/"inputs"/"instances.json").read_text())["graphs"]
    for row in rows:
        cut = int(row["cut"])
        order = row["order"]
        if row["instance"] == "copy2" and row["tensor"] == "joint_gibbs":
            require(order in {"grouped", "interleaved"}, row)
            expected = copy_rank(order, cut)
            require(int(row["exact_rank"]) == expected, row)
            if order == "grouped" and cut == 2:
                # Eigenvectors of the pair factor are (1,+/-1).
                # The squared tail for r=3 is exactly .01 relative to the total.
                require(int(row["eta_0p1"]) == 3 and int(row["eta_0p01"]) == 4, row)
            copy[(order, cut)] = expected
        if row["instance"] in graphs and row["tensor"] in {"objective","gibbs"}:
            name, transform, tensor = row["instance"], row["map"], row["tensor"]
            spec = graphs[name]
            labels = spec["orders"][order]
            require(labels == [int(i) for i in row["order_labels"].split(",")], row)
            expected = graph_rank(spec, labels, transform, cut, tensor == "gibbs")
            require(int(row["exact_rank"]) == expected, row)
            key = (name,order,transform,tensor,cut)
            require(key not in graph_rows, ("duplicate graph row",key))
            graph_rows[key] = expected
    require(len(copy) == 6, ("copy rank coverage", copy))
    counts = {name:sum(k[0]==name for k in graph_rows) for name in graphs}
    # The published smoke mode includes only the triangle.
    require(counts["triangle"] == 16, ("triangle coverage", counts))
    if counts.get("tree") or counts.get("square"):
        for name, spec in graphs.items():
            maps = 3 if "parents" in spec else 2
            require(counts[name] == 2 * (spec["n"]-1) * len(spec["orders"]) * maps,
                    ("graph coverage",name,counts))
    return {"copy_unfoldings": len(copy), "graph_unfoldings": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    report = {
        "original_objectives": check_originals(),
        "mixed_pointwise_assignments": check_mixed(table(results/"mixed_points.csv")),
        "tv_as_exact_fractions": check_gibbs(table(results/"gibbs.csv")),
        **check_ranks(table(results/"ranks.csv")),
        "rebuilt_file_sha256": check_rebuild(results),
    }
    report["status"] = "passed"
    data = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output or results/"independent_check.json").write_text(data)
    print(data, end="")


if __name__ == "__main__":
    main()
