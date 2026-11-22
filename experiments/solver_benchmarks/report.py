"""Rebuild fixed-budget summaries and plots from recorded solver runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FORMS = ("quadratic", "xor", "original", "directional", "local")
PAIRS = (("quadratic", "xor"), ("original", "directional"),
         ("original", "local"), ("local", "directional"))
COLORS = dict(zip(FORMS, ("#0072b2", "#d55e00", "#0072b2", "#009e73", "#cc79a7")))
GROUP = ("case", "solver", "budget")
FORM_HEADER = (*GROUP, "form", "n", "score_median", "score_q25", "score_q75",
               "returned_score_median", "best_hits", "best_hit_rate", "returned_hits",
               "returned_hit_rate", "returned_feasible_rate", "seconds_median",
               "seconds_q25", "seconds_q75", "optimum", "maximize")
PAIR_HEADER = (*GROUP, "reference", "alternative", "seed", "ref_score", "alt_score",
               "gain", "returned_gain", "seconds_difference")
CONTRAST_HEADER = (*GROUP, "reference", "alternative", "n", "gain_median", "gain_q25",
                   "gain_q75", "gain_min", "gain_max", "improved", "tied", "worse",
                   "returned_gain_median", "seconds_difference_median")
CURVE_HEADER = (*GROUP, "form", "n", "evaluations", "score_median", "score_q25", "score_q75")


def dump(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, q):
    """Linear quantile at q*(n-1), including singleton groups."""
    data = sorted(values)
    pos = (len(data) - 1) * q
    lo = math.floor(pos)
    return data[lo] + (pos - lo) * (data[math.ceil(pos)] - data[lo])


def read_runs(root):
    valid = defaultdict(dict)
    coverage = []
    seen = set()
    for file in sorted((root / "runs").glob("*.json")):
        rec = json.loads(file.read_text(encoding="utf-8"))
        group = tuple(rec[k] for k in GROUP)
        form, seed = rec["form"], rec["seed"]
        key = (*group, form, seed)
        if key in seen:
            raise ValueError(f"Duplicate instance/solver/budget/form/seed: {file.name}")
        seen.add(key)
        status = ("error" if rec["exit_code"] != 0 else
                  "unequal_budget" if rec["n_evals"] != rec["budget"] else "included")
        coverage.append(dict(run_id=rec.get("run_id", file.stem),
                             **dict(zip(GROUP, group)), form=form, seed=seed,
                             n_evals=rec["n_evals"], exit_code=rec["exit_code"], status=status))
        if status == "included":
            trace = rec["trace"]
            if (not trace or trace[0][0] != 1 or trace[-1][1] != rec["best_original"]["score"]
                    or any(not 1 <= step <= rec["budget"] for step, _ in trace)):
                raise ValueError(f"Invalid recorded trace in {file.name}")
            for (step1, score1), (step2, score2) in zip(trace, trace[1:]):
                if step1 >= step2 or (score2 - score1) * (1 if rec["maximize"] else -1) < 0:
                    raise ValueError(f"Nonmonotone best-so-far trace in {file.name}")
            valid[group][form, seed] = rec
    if not coverage:
        raise ValueError(f"No run records in {root / 'runs'}")
    return valid, coverage


def summarize(valid):
    forms, paired, contrasts, curves = [], [], [], []
    for group, members in sorted(valid.items()):
        key = dict(zip(GROUP, group))
        any_run = next(iter(members.values()))
        optimum, maximize = any_run["optimum"], any_run["maximize"]
        if any((r["optimum"], r["maximize"]) != (optimum, maximize) for r in members.values()):
            raise ValueError(f"Inconsistent original objective metadata within {group}")
        sign = 1 if maximize else -1
        for form in FORMS:
            reps = [r for (f, _), r in members.items() if f == form]
            if not reps:
                continue
            scores = [r["best_original"]["score"] for r in reps]
            returned = [r["returned"]["score"] for r in reps]
            secs = [r["seconds"] for r in reps]
            hits = sum(s == optimum for s in scores) if optimum is not None else None
            hits_ret = sum(s == optimum for s in returned) if optimum is not None else None
            n = len(reps)
            forms.append(dict(**key, form=form, n=n, score_median=median(scores),
                              score_q25=quantile(scores, .25), score_q75=quantile(scores, .75),
                              returned_score_median=median(returned), best_hits=hits,
                              best_hit_rate=hits/n if hits is not None else None,
                              returned_hits=hits_ret, returned_hit_rate=hits_ret/n if hits_ret is not None else None,
                              returned_feasible_rate=sum(r["returned"]["feasible"] for r in reps)/n,
                              seconds_median=median(secs), seconds_q25=quantile(secs, .25),
                              seconds_q75=quantile(secs, .75), optimum=optimum, maximize=maximize))
            # Each trace is a right-continuous step function. The union of actual
            # improvement times recovers all changes of its pointwise quantiles.
            traces = [list(zip(*r["trace"])) for r in reps]
            steps = sorted({key["budget"]}.union(*(indexes for indexes, _ in traces)))
            for step in steps:
                values = [scores_at[bisect_right(indexes, step)-1] for indexes, scores_at in traces]
                curves.append(dict(**key, form=form, n=n, evaluations=step, score_median=median(values),
                                   score_q25=quantile(values, .25), score_q75=quantile(values, .75)))
        for ref, alt in PAIRS:
            seeds = sorted({s for f, s in members if f == ref} & {s for f, s in members if f == alt})
            if not seeds:
                continue
            rows = []
            for seed in seeds:
                a, b = members[ref, seed], members[alt, seed]
                score_a = a["best_original"]["score"]
                score_b = b["best_original"]["score"]
                rows.append(dict(**key, reference=ref, alternative=alt, seed=seed, ref_score=score_a,
                                 alt_score=score_b, gain=sign*(score_b-score_a),
                                 returned_gain=sign*(b["returned"]["score"]-a["returned"]["score"]),
                                 seconds_difference=b["seconds"]-a["seconds"]))
            paired.extend(rows)
            gain = [r["gain"] for r in rows]
            contrasts.append(dict(**key, reference=ref, alternative=alt, n=len(rows), gain_median=median(gain),
                                  gain_q25=quantile(gain, .25), gain_q75=quantile(gain, .75),
                                  gain_min=min(gain), gain_max=max(gain), improved=sum(x > 0 for x in gain),
                                  tied=gain.count(0), worse=sum(x < 0 for x in gain),
                                  returned_gain_median=median(r["returned_gain"] for r in rows),
                                  seconds_difference_median=median(r["seconds_difference"] for r in rows)))
    return forms, paired, contrasts, curves


def panel_plots(out, valid, curves, paired):
    cases = sorted({k[0] for k in valid})
    columns = sorted({k[1:] for k in valid})
    nrow, ncol = len(cases), len(columns)
    f1, grid1 = plt.subplots(nrow, ncol, figsize=(3.9*ncol, 2.05*nrow+1.15), squeeze=False)
    f2, grid2 = plt.subplots(nrow, ncol, figsize=(4.15*ncol, 1.95*nrow+1.1), squeeze=False)
    for i, case in enumerate(cases):
        for j, (solver, budget) in enumerate(columns):
            axis, delta_ax = grid1[i, j], grid2[i, j]
            group = (case, solver, budget)
            if group not in valid:
                axis.set_visible(False)
                delta_ax.set_visible(False)
                continue
            run = next(iter(valid[group].values()))
            title = f"{case} · {solver} · {budget} calls"
            maximize = run["maximize"]
            for ax in (axis, delta_ax):
                ax.set_title(title, loc="left", fontsize=9)
                ax.spines[["right", "top"]].set_visible(False)
                ax.tick_params(labelsize=8)
            axis.set_ylabel("Cut weight ↑" if maximize else "HUBO energy ↓", fontsize=8)
            axis.set_xlabel("Objective evaluations", fontsize=8)
            subset = [r for r in curves if tuple(r[k] for k in GROUP) == group]
            for form, items in groupby(subset, key=lambda r: r["form"]):
                rows = list(items)
                x = [r["evaluations"] for r in rows]
                axis.step(x, [r["score_median"] for r in rows], where="post", color=COLORS[form], lw=1.5)
                axis.fill_between(x, [r["score_q25"] for r in rows], [r["score_q75"] for r in rows],
                                  color=COLORS[form], alpha=.12, linewidth=0, step="post")
            axis.set_xlim(1, budget)
            if run["optimum"] is not None:
                axis.axhline(run["optimum"], ls=":", color="#505050", lw=.9)
            rows = [r for r in paired if tuple(r[k] for k in GROUP) == group]
            delta_ax.axvline(0, color="#777777", lw=.85)
            labels = []
            for pair, points in groupby(rows, key=lambda r: (r["reference"], r["alternative"])):
                points = list(points)
                idx = len(labels)
                labels.append(f"{pair[1]} vs {pair[0]}")
                gains = [p["gain"] for p in points]
                # Deterministic vertical offsets make repeated paired differences visible.
                y = [idx + .36*((k+.5)/len(gains)-.5) for k in range(len(gains))]
                delta_ax.scatter(gains, y, s=14, color=COLORS[pair[1]], alpha=.75)
                delta_ax.plot(median(gains), idx, marker="|", ms=12, color="black")
            delta_ax.set_yticks(range(len(labels)), labels, fontsize=7)
            delta_ax.set_ylim(-.55, max(.55, len(labels)-.45))
            delta_ax.set_xlabel("Gain in original objective units (positive = better)", fontsize=7)
    handles = [Line2D([0], [0], color=COLORS[f], label=f) for f in FORMS]
    handles.append(Line2D([0], [0], color="#505050", ls=":", label="optimum"))
    f1.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.5,.97), frameon=False, fontsize=9)
    f1.suptitle("Best decoded score across seeds · median and interquartile range", y=.994, fontsize=12)
    f2.suptitle("Matched-seed gains · dots are seeds, bars are medians", y=.994, fontsize=12)
    for fig, name, top in ((f1, "best_so_far", .935), (f2, "paired_gains", .96)):
        fig.tight_layout(rect=(0, 0, 1, top), h_pad=1.1)
        for extension in ("pdf", "svg"):
            meta = {"CreationDate": None, "ModDate": None} if extension == "pdf" else {"Date": None}
            fig.savefig(out / f"{name}.{extension}", bbox_inches="tight", metadata=meta)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/derived"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    valid, coverage = read_runs(args.results)
    dump(args.out/"run_coverage.csv", tuple(coverage[0]), coverage)
    forms, paired, contrasts, curves = summarize(valid)
    for name, header, rows in (("by_form", FORM_HEADER, forms), ("paired_seeds", PAIR_HEADER, paired),
                               ("paired_summary", CONTRAST_HEADER, contrasts), ("trajectories", CURVE_HEADER, curves)):
        dump(args.out/f"{name}.csv", header, rows)
    if valid:
        matplotlib.rcParams["svg.hashsalt"] = "solver-benchmark-v1"
        panel_plots(args.out, valid, curves, paired)
    print(f"{len(coverage)} observed runs; {sum(r['n'] for r in forms)} included; {len(paired)} matched pairs.")


if __name__ == "__main__":
    main()
