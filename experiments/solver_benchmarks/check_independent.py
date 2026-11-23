"""Audit the benchmark using gate truth tables and direct weighted cut scores."""
import argparse
import base64
import csv
import hashlib
import json
import math
import statistics
import zlib
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def bits(d):
    return product(range(2), repeat=d)


def gate_charge(a, b, u, v, w):
    # Rows are the number of true parents; columns are 0 or 1 at the gate.
    return ((0, a+2*b), (0, b), (a, 0))[u+v][w]


def mixed_original(x):
    return x[0]*x[1]*x[2]*(2*x[3] - 3*x[4])


def mixed_energy(z, a, b):
    p,q,r,i,j,w,t = z
    return t*(2*i-3*j) + gate_charge(a,b,p,q,w) + gate_charge(a,b,w,r,t)


def coefficients(values, dimension):
    """Boolean lattice inversion; bit zero is the first variable."""
    out = list(values)
    for i in range(dimension):
        for mask in range(2**dimension):
            if mask & (1 << i):
                out[mask] -= out[mask ^ (1 << i)]
    return {mask: val for mask, val in enumerate(out) if val}


def precision(output):
    rows = []
    original = {x: mixed_original(x) for x in bits(5)}
    opt = {x for x in original if original[x] == min(original.values())}
    masks = [tuple((mask >> i) & 1 for i in range(7)) for mask in range(128)]
    for policy in ("directional", "local"):
        settings = [(Fraction(1), d) for d in map(Fraction, (0, "1/4", 1, 4, 16))]
        settings.append((Fraction(1,2), Fraction(0)))
        for scale, delta in settings:
            a = (2 if policy == "directional" else 3)*scale + delta
            b = 3*scale + delta
            coeff = coefficients([mixed_energy(z,a,b) for z in masks], 7)
            require(all(bin(mask).count("1") <= 2 for mask in coeff), "QUBO reconstruction")
            magnitudes = [abs(val) for mask,val in coeff.items() if mask]
            bound = max(magnitudes)
            for width in (0, 3, 5, 8):
                # The symmetric signed levels include zero and both extrema.
                step = bound / (2**(width-1)-1) if width else Fraction(0)
                vals = {mask: step*round(val/step) if width else val for mask, val in coeff.items()}
                fibers = defaultdict(list)
                for mask,z in enumerate(masks):
                    energy = sum(val for term,val in vals.items() if (mask & term) == term)
                    if not width:
                        require(energy == mixed_energy(z, a, b), "round-trip polynomial")
                    fibers[z[:5]].append((energy,z[5:]))
                f_min, ties, false_better = {}, 0, 0
                for x, alternatives in fibers.items():
                    f_min[x] = min(e for e,_ in alternatives)
                    consistent = (x[0]*x[1], math.prod(x[:3]))
                    allowed = {y for e,y in alternatives if e == f_min[x]}
                    ties += bool(allowed - {consistent})
                    false_better += consistent not in allowed
                wrong = sum(original[x] != e for x,e in f_min.items())
                new_opt = {x for x,e in f_min.items() if e == min(f_min.values())}
                if not width and scale == 1:
                    require(wrong == 0 and (delta == 0 or ties == 0), (policy,delta))
                rows.append(dict(policy=policy, load_scale=str(scale), margin=str(delta),
                                 signed_bits=width or "unrounded", max_abs_coefficient=str(bound),
                                 min_abs_coefficient=str(min(magnitudes)), quantum=str(step),
                                 mismatched_fibers=wrong, fiber_min_equivalent=int(wrong == 0),
                                 max_abs_fiber_error=str(max(abs(e-original[x]) for x,e in f_min.items())),
                                 fibers_with_inconsistent_minimizer=ties,
                                 fibers_without_consistent_minimizer=false_better,
                                 decoded_optima_count=len(new_opt),
                                 changed_optimum_assignments=len(opt.symmetric_difference(new_opt))))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"cases": len(rows), "fibers_per_case": len(original), "states_per_case": len(masks),
            "pointwise_failures": sum(not r["fiber_min_equivalent"] for r in rows)}


def direct_cut(x, edges):
    return sum(w for i,j,w in edges if x[i] != x[j])


def regression(control):
    star = control["star_regression"]
    good, bad = [], []
    for x in bits(star["n"]):
        center, *leaves = x
        good.append(direct_cut(x, star["edges"]))
        bad.append(sum(leaves) - center - 2*math.prod(leaves))
    require(max(good) == star["cut_max"] == 3 and max(bad) == star["invalid_surrogate_max"] == 2,
            "V.4 counterexample")
    return {"cut_max":max(good), "surrogate_max":max(bad), "pointwise_mismatches":sum(a!=b for a,b in zip(good,bad))}


def write_report(path, report):
    data = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(data)
    print(data, end="")


def shared_controls(control):
    triangle = control["triangle"]
    u,v = triangle["u"], triangle["v"]
    scores = []
    for z in bits(triangle["n"]):
        x = list(z)
        x[u] ^= x[v]
        scores.append(direct_cut(x, triangle["edges"]))
    require(scores == triangle["xor_scores"] and max(scores) == triangle["maximum"], "triangle scores")

    mix = control["mixed"]
    require(mix["shared_gates"] == [[0,1],[5,2]], "mixed gate order")
    stats = {}
    z_points = [tuple((mask >> i) & 1 for i in range(7)) for mask in range(128)]
    for policy in ("directional","local","global"):
        setting = mix[policy]
        a0,a1 = setting["A"]
        b0,b1 = setting["B"]
        require(a0==a1 and b0==b1, policy)
        energies = [mixed_energy(z, a0, b0) for z in z_points]
        coeff = [abs(c) for mask,c in coefficients(energies, 7).items() if mask]
        require(min(coeff) == setting["min_abs"] and max(coeff) == setting["max_abs"], policy)
        for x in bits(5):
            values = [mixed_energy((*x,*y),a0,b0) for y in bits(2)]
            require(min(values) == mixed_original(x) and values.count(min(values)) == 1, x)
        stats[policy] = {"states":len(energies), "min_abs":min(coeff), "max_abs":max(coeff)}
    return {"triangle_scores":scores, "shared_mixed":stats, "invalid_substitution":regression(control)}


def fixture_agreement(control, graph_dir):
    import numpy as np
    from models import load
    triangle = load("triangle",graph_dir)
    triangle_points = list(bits(control["triangle"]["n"]))
    cut_scores = -triangle.evaluate("xor",np.asarray(triangle_points))
    require(list(cut_scores)==control["triangle"]["xor_scores"], "B2/B3 triangle truth table")
    mixed = load("mixed",graph_dir)
    mix = control["mixed"]
    x_points = list(bits(mix["n"]))
    require(list(mixed.evaluate("original",np.asarray(x_points)))==[mixed_original(x) for x in x_points],
            "B2/B3 mixed original values")
    z_points = list(bits(7))
    for policy in ("directional","local"):
        a,b = mix[policy]["A"][0], mix[policy]["B"][0]
        require(list(mixed.evaluate(policy,np.asarray(z_points))) ==
                [mixed_energy(z,a,b) for z in z_points], ("B2/B3 mixed losses",policy))
    return {"triangle_xor_values":len(triangle_points), "mixed_original_values":len(x_points),
            "mixed_directional_and_local_energies":2*len(z_points)}


class Oracle:
    """Reconstruct inputs and penalty loads from the term prefixes, without compiling a QUBO."""

    def __init__(self, model):
        self.n = model.n
        self.maximize = model.maximize
        self.optimum = model.optimum
        self.edges = tuple(getattr(model, "edges", ()) or ())
        if self.edges:
            self.u, self.v = model.u, model.v
            return
        self.delta = model.delta
        self.terms = list(model.terms)
        prefixes = {}
        self.gates, self.loads, self.readout = [], [], []
        for weight, support in self.terms:
            ordered = tuple(sorted(support))
            for k in range(2,len(ordered)):
                prefix = ordered[:k]
                if prefix not in prefixes:
                    g = len(self.gates)
                    prefixes[prefix] = g
                    left = ordered[0] if k==2 else self.n + prefixes[prefix[:-1]]
                    self.gates.append((left, ordered[k-1]))
                    self.loads.append([0,0])
                g = prefixes[prefix]
                self.loads[g][0] += max(weight,0)
                self.loads[g][1] += max(-weight,0)
            cut = (self.n + prefixes[ordered[:-1]], ordered[-1]) if len(ordered)>2 else ordered
            self.readout.append((weight,cut))
        require([tuple(g) for g in model.lift.gates] == self.gates, "compiled gate order")
        require([list(load) for load in model.lift.loads] == self.loads, "compiled signed loads")

    def evaluate(self, form, candidate):
        z = tuple(int(i) for i in candidate)
        require(all(i in (0,1) for i in z) and all(a==b for a,b in zip(z,candidate)), candidate)
        x = z[:self.n]
        if self.edges:
            require(len(z)==self.n and form in ("quadratic","xor"), (form,z))
            x = list(x)
            if form=="xor":
                x[self.u] ^= x[self.v]
            score = direct_cut(x, self.edges)
            return tuple(x), int(score), True, int(-score)
        score = sum(c*math.prod(x[i] for i in s) for c,s in self.terms)
        if form=="original":
            require(len(z)==self.n, (form,z))
            return x, int(score), True, int(score)
        require(form in ("directional","local") and len(z)==self.n+len(self.gates), (form,z))
        loss = sum(c*math.prod(z[i] for i in s) for c,s in self.readout)
        bad = 0
        for g,((i,j),(p,nload)) in enumerate(zip(self.gates,self.loads)):
            a,b = p+self.delta,nload+self.delta
            if form == "local":
                a = b = max(a,b)
            y = z[self.n+g]
            bad += y != z[i]*z[j]
            loss += gate_charge(a,b,z[i],z[j],y)
        require(loss-score >= self.delta*bad, ("penalty certificate",z))
        return x, int(score), not bad, int(loss)

    def consistent(self, x):
        z = list(x)
        for i,j in self.gates:
            z.append(z[i]*z[j])
        return tuple(z)


def all_large_fibers(model, oracle, form):
    import numpy as np
    n, d = oracle.n, model.dimension(form)
    size = 1 << (d-n)
    # Lexicographic ordering makes all extensions of x contiguous.
    checked = fibers = 0
    shifts = np.arange(d-1,-1,-1)
    for start in range(0,1<<d,8192):
        ids = np.arange(start,min(1<<d,start+8192))
        z = ((ids[:,None] >> shifts) & 1).astype(np.uint8)
        score,loss = batch_oracle(z,oracle,form)
        require(np.array_equal(loss,model.evaluate(form,z)),("full lifted energies",model.case,form,start))
        fiber_losses = loss.reshape(-1,size)
        original = score[::size,None]
        require(np.all(fiber_losses.min(axis=1)==original[:,0]) and
                np.all(np.sum(fiber_losses==original,axis=1)==1), ("full lifted minima",model.case,form,start))
        checked += len(z)
        fibers += len(original)
    return checked,fibers


def preflight_models(graph_dir):
    import itertools
    import random
    import numpy as np
    from models import CASE_IDS, formulations, load

    report = {}
    for case in CASE_IDS:
        model = load(case, graph_dir)
        oracle = Oracle(model)
        stats = {}
        for form in formulations(case):
            dimension = model.dimension(form)
            exhaust = dimension <= 16
            if exhaust:
                points = bits(dimension)
            else:
                # Fixed sample per dimension/case, reused across same-dimensional formulations.
                rng = random.Random("B3 audit v1/" + case + "/" + str(dimension))
                points = [tuple([value]*dimension) for value in (0,1)]
                points += [tuple(rng.getrandbits(1) for _ in range(dimension)) for _ in range(128)]
                if not oracle.edges and form != "original":
                    for _ in range(32):
                        x = tuple(rng.getrandbits(1) for _ in range(model.n))
                        z = oracle.consistent(x)
                        points.append(z)
                        for g in range(len(oracle.gates)):
                            flipped = list(z)
                            flipped[model.n+g] ^= 1
                            points.append(tuple(flipped))
            points = iter(points)
            checked, fibers = 0, {}
            optimal = None
            while True:
                batch = list(itertools.islice(points, 512))
                if not batch:
                    break
                vals = model.evaluate(form, np.asarray(batch,dtype=np.int64))
                require(len(vals) == len(batch), ("vectorized output length",case,form))
                for z, value in zip(batch, vals):
                    x,score,ok,loss = oracle.evaluate(form,z)
                    decode,raw_score,raw_ok = model.decode(form,z)
                    require((tuple(decode),raw_score,bool(raw_ok),value) == (x,score,ok,loss),
                            ("production value/decoder",case,form,z))
                    if exhaust and form in ("directional","local"):
                        best,count,original = fibers.get(x,(math.inf,0,score))
                        fibers[x] = (min(loss,best), (count+1 if loss==best else (1 if loss<best else count)), original)
                    if optimal is None or (score > optimal if model.maximize else score < optimal):
                        optimal = score
                    checked += 1
            if exhaust and model.optimum is not None:
                require(optimal == model.optimum, ("known optimum",case,form,optimal,model.optimum))
            if fibers:
                require(len(fibers)==2**model.n and all(lo==f and count==1 for lo,count,f in fibers.values()),
                        ("pointwise minimum or uniqueness",case,form))
            stats[form] = {"exhaustive":exhaust,"assignments":checked,"exact_fibers":len(fibers),
                           "checked_native_decodes":checked}
            if not exhaust and not oracle.edges and form in ("directional","local"):
                count, num_fibers = all_large_fibers(model,oracle,form)
                stats[form].update(exhaustive=True,assignments=count,exact_fibers=num_fibers)
        report[case] = stats
    return report


def verify_candidate(record, oracle, form):
    raw = record["bits"]
    z = list(map(int,raw)) if isinstance(raw,str) else raw
    x, score, ok, loss = oracle.evaluate(form,z)
    require((list(map(int,record["decoded"])),record["score"],record["feasible"],record["loss"]) ==
            (list(x),score,ok,loss), ("stored candidate", record))
    return score


def batch_oracle(z, oracle, form):
    import numpy as np
    count = len(z)
    # Contiguous row order is preserved by the packing. For graph scores the
    # XOR below is decoded before comparing edge endpoints.
    if oracle.edges:
        x = z.copy()
        if form == "xor":
            x[:,oracle.u] ^= x[:,oracle.v]
        score = np.zeros(count,dtype=np.int64)
        for i,j,w in oracle.edges:
            score += w*(x[:,i]!=x[:,j])
        loss = -score
    else:
        def readout(terms):
            vals = np.zeros(count,dtype=np.int64)
            for c,s in terms:
                vals += c*np.prod(z[:,s],axis=1,dtype=np.int64)
            return vals
        score = readout(oracle.terms)
        loss = score
        if form != "original":
            loss = readout(oracle.readout)
            bad = np.zeros(count,dtype=np.int64)
            for g,((i,j),(p,nload)) in enumerate(zip(oracle.gates,oracle.loads)):
                a,b = p+oracle.delta, nload+oracle.delta
                if form == "local":
                    a = b = max(a,b)
                bad += z[:,oracle.n+g] != z[:,i]*z[:,j]
                parent_count = z[:,i]+z[:,j]
                child = z[:,oracle.n+g]
                loss += np.asarray([[0,a+2*b],[0,b],[a,0]])[parent_count,child]
            require(np.all(loss-score >= oracle.delta*bad), "independent violation-pattern certificate")
    return score,loss


def audit_points(record, oracle):
    import numpy as np
    count, d = record["n_evals"], record["dimension"]
    def unpacked(field, dtype):
        data = zlib.decompress(base64.b64decode(record[field], validate=True))
        return np.frombuffer(data, dtype=dtype)
    packed = unpacked("eval_bits_b64", np.uint8).reshape(count,(d+7)//8)
    whole = np.unpackbits(packed,axis=1,bitorder="big")
    require(not np.any(whole[:,d:]), ("padding bits",record["run_id"]))
    z = whole[:,:d]
    saved = unpacked("eval_loss_b64","<i4")
    require(saved.shape==(count,), ("stored losses",record["run_id"]))

    score,loss = batch_oracle(z,oracle,record["form"])
    require(np.array_equal(saved,loss), ("pointwise logged losses",record["run_id"]))
    for field in ("best_original","returned"):
        target = record[field]
        z_target = list(map(int,target["bits"]))
        matches = np.flatnonzero(np.all(z==z_target,axis=1))
        index = target["eval_index"]-1
        require(index in matches and np.all(loss[matches]==target["loss"]),(field,record["run_id"]))
    require(record["n_distinct"] == len(np.unique(packed,axis=0)), ("distinct states",record["run_id"]))
    require(record["returned"]["loss"] == int(loss.min()), ("native return",record["run_id"]))
    ordered = score if oracle.maximize else -score
    best = np.maximum.accumulate(ordered)
    changed = np.r_[True,best[1:]>best[:-1]]
    trace = [[int(i+1),int(score[i])] for i in np.flatnonzero(changed)]
    require(trace == record["trace"] and trace[-1][1]==record["best_original"]["score"],
            ("full recorded trajectory",record["run_id"]))
    return count


def check_records(results, graph_dir):
    from models import CASE_IDS, formulations, load
    design = json.loads((results/"design.json").read_text())
    if design["profile"] == "full":
        cases, seeds, planned_budget = CASE_IDS, range(201,209), 4096
    elif design["profile"] == "smoke":
        cases, seeds, planned_budget = ("triangle","mixed"), (200,), 128
    else:
        raise ValueError(design["profile"])
    expected = {(case,solver,form,planned_budget,seed) for case in cases
                for solver in ("PROTES","TTOpt") for form in formulations(case) for seed in seeds}
    require(design["budget"] == planned_budget and design["seeds"] == list(seeds), "design limits")
    records = sorted((results/"runs").glob("*.json"))
    require(records, "no completed run records")
    oracles = {}
    groups = defaultdict(list)
    n_ok = checked_points = 0
    run_ids, conditions = set(), set()
    for path in records:
        r = json.loads(path.read_text())
        case,solver,form = r["case"],r["solver"],r["form"]
        budget, n = r["budget"],r["n_evals"]
        rid = r["run_id"]
        require(rid == path.stem and rid not in run_ids, ("non-unique run_id",path))
        run_ids.add(rid)
        key = (case,solver,form,budget,r["seed"])
        require(key not in conditions, ("duplicate condition",key))
        conditions.add(key)
        require(r["code_sha"] == design["code_sha"], ("code revision",rid))
        if r["exit_code"] != 0:
            continue
        if case not in oracles:
            oracles[case] = Oracle(load(case,graph_dir))
        oracle = oracles[case]
        batches = r["batches"]
        require(all(isinstance(b,int) and b>0 for b in batches) and sum(batches) == n and n<=budget,
                ("accounting",rid))
        require(n == budget and math.isfinite(r["seconds"]) and r["seconds"] >= 0, ("completed run",rid))
        checked_points += audit_points(r,oracle)
        returned = verify_candidate(r["returned"],oracle,form)
        best = verify_candidate(r["best_original"],oracle,form)
        sign = 1 if oracle.maximize else -1
        require(sign*best >= sign*returned, ("incumbents",rid))
        points = r["trace"]
        require(points and points[0][0] == 1 and points[-1][1] == best, ("trace endpoints",rid))
        last_index, last_score = 0, -math.inf
        for i,s in points:
            require(last_index < i <= n and last_score < sign*s, ("trace order",rid))
            last_index, last_score = i,sign*s
        if n == budget and r["exit_code"] == 0:
            groups[(case,solver,budget,form)].append(best)
            n_ok += 1

    require(conditions == expected and set(design["run_order"]) == run_ids and
            len(design["run_order"]) == len(expected), "scheduled result coverage")
    with (results/"derived"/"by_form.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    reported = set()
    for row in rows:
        key = (row["case"],row["solver"],int(row["budget"]),row["form"])
        require(key not in reported, ("repeated aggregate",key))
        reported.add(key)
        scores = groups[key]
        require(len(scores)==int(row["n"]) and statistics.median(scores)==float(row["score_median"]),
                ("independent median",key,scores,row))
    require(set(groups)==reported, ("summary coverage", set(groups)^reported))
    return {"status":"passed","records":len(records),"successful_equal_budget_runs":n_ok,
            "natural_score_medians_reproduced":len(reported),
            "candidate_checks":["decoded bits","original score","feasibility","minimization loss"],
            "trajectory_checks":["strict improvements","evaluation coordinates","terminal decoded best"],
            "batch_counts_reconciled":n_ok, "decoded_evaluations_rescored":checked_points}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight","postrun"))
    parser.add_argument("--graphs", type=Path, help="Directory containing checksum-verified benchmark graph files")
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path, help="Override the JSON report path")
    args = parser.parse_args()
    args.results.mkdir(parents=True,exist_ok=True)
    if args.mode == "preflight":
        control_path = HERE/"inputs"/"controls-v1.json"
        control_sha = hashlib.sha256(control_path.read_bytes()).hexdigest()
        expected = json.loads((HERE/"inputs"/"models.json").read_text())["b2"]["origin_files"]
        require(control_sha == expected["inputs/controls-v1.json"], "B2 fixture digest")
        report = {"b2_controls_sha256":control_sha,
                  "shared_controls":shared_controls(json.loads(control_path.read_text())),
                  "b2_to_production":fixture_agreement(json.loads(control_path.read_text()),args.graphs),
                  "production":preflight_models(args.graphs), "status":"passed"}
        target = args.results/"independent_preflight.json"
    else:
        report = check_records(args.results, args.graphs)
        precision_output = args.output.with_suffix(".precision.csv") if args.output else args.results / "precision.csv"
        report["penalty_precision"] = precision(precision_output)
        target = args.results/"independent.json"
    write_report(args.output or target,report)


if __name__ == "__main__":
    main()
