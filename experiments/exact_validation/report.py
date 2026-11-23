"""Rebuild all derived tables and vector figures from the raw CSV files."""
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import os

os.environ.setdefault("MPLBACKEND","Agg")
import matplotlib as mpl
mpl.use("Agg",force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

mpl.rcParams.update({"svg.hashsalt":"exact-validation-v1","font.size":9,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.grid":True,"grid.alpha":.22,
                     "figure.constrained_layout.use":True})


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(rows,path):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as stream:
        w = csv.DictWriter(stream,keys,lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def figures(fig,root,stem):
    fig.savefig(root/(stem+".svg"), metadata={"Date":None})
    fig.savefig(root/(stem+".pdf"), metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


def rebuild(raw,out):
    out.mkdir(exist_ok=True,parents=True)
    product = read(raw/"products.csv")
    compare = []
    for row in product:
        if row["penalty"]=="directional" and row["delta"]=="1":
            group = {r["penalty"]:r for r in product if (r["model"],r["scheme"],r["delta"])==
                     (row["model"],row["scheme"],"1")}
            summary = {k:row[k] for k in ("model","scheme","m","minimum")}
            for kind,r in group.items():
                summary[kind+"_max"] = r["max_abs"]
                summary[kind+"_ratio"] = r["coefficient_ratio"]
            compare.append(summary)
    save(compare,out/"penalty_table.csv")
    gibbs = read(raw/"gibbs.csv")
    keys = ("family","m","delta","beta","M","tv","b1","b2")
    focus = [r for r in gibbs if r["m"]=="2" and r["delta"]=="1" and abs(float(r["beta"])-.6931471805599453)<1e-12]
    save([{k:r[k] for k in keys} for r in focus],out/"gibbs_table.csv")
    fig,(ax,bx) = plt.subplots(1,2,figsize=(8.0,3.05))
    ordinary = [r for r in gibbs if r["family"]=="ordinary" and r["m"]=="2" and r["delta"]=="1"]
    sharp = [r for r in focus if r["family"]=="sharpness"]
    for axis,rows,key,xlab,title in [(ax,ordinary,"beta",r"Inverse temperature $\beta$","Shared product lift"),
                                     (bx,sharp,"M",r"Interaction increment $M$","Finite sharpness family")]:
        for y,label,sty,color in [("tv","Exact TV","o-","#174d6f"),("b1","Tanh bound","s--","#ae5200"),
                                  ("b2","Exponential bound","^:","#5a3982")]:
            axis.plot([float(r[key]) for r in rows],[float(r[y]) for r in rows],sty,label=label,color=color,ms=4)
        axis.set(xlabel=xlab,title=title,ylim=(-.011,None))
    ax.set_ylabel("Total variation distance or bound")
    bx.legend(fontsize=8,loc="lower right")
    figures(fig,out,"gibbs_bounds")
    ranks = read(raw/"ranks.csv")
    groups = defaultdict(dict)
    for r in ranks:
        groups[(r["instance"],r["order"])].setdefault(r["map"],[]).append(r)
    selection = []
    for (instance,order),maps in groups.items():
        if not {"identity","xor"}<=set(maps):
            continue
        scores = {}
        for name,rows in maps.items():
            obj = [r for r in rows if r["tensor"]=="objective"]
            gib = [r for r in rows if r["tensor"]=="gibbs"]
            scores[name] = (max(int(r["eta_0p01"]) for r in gib),max(int(r["exact_rank"]) for r in obj),
                            int(obj[0]["degree"]),max(int(r["kappa"]) for r in obj),name)
        winner = min(scores,key=scores.get)
        for name,(gib,obj,degree,kmax,_) in scores.items():
            selection.append(dict(instance=instance,order=order,map=name,max_gibbs_rank_1pct=gib,
                                  max_objective_rank=obj,degree=degree,max_kappa=kmax,selected=int(name==winner)))
    save(selection,out/"coordinate_table.csv")
    case = "tree" if ("tree","natural") in groups else "triangle"
    fig,(ax,bx) = plt.subplots(1,2,figsize=(8,3.05))
    for axis,tensor,field,title in [(ax,"objective","exact_rank","Exact objective rank"),
                                    (bx,"gibbs","eta_0p01","Gibbs rank at 1% error")]:
        for name,rows in groups[(case,"natural")].items():
            points = [r for r in rows if r["tensor"]==tensor]
            axis.plot([int(r["cut"]) for r in points],[int(r[field]) for r in points],marker={"identity":"o","xor":"s","tree_edge":"^"}[name],
                      label=name.replace("_"," "))
        axis.set(xlabel="Cut after variable count",ylabel="Unfolding rank",title=title)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_ylim(bottom=.7)
    bx.legend(fontsize=8)
    figures(fig,out,"coordinate_ranks")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=Path("results"))
    p.add_argument("--out",type=Path,default=Path("results/derived"))
    a = p.parse_args()
    rebuild(a.results,a.out)
