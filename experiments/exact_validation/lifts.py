"""Batched exhaustive integer fibers and exact finite Gibbs marginals."""
from math import exp, log, log1p, tanh
import numpy as np

from exact_api import bits, poly_value, product_lift


def word(x):
    return "".join(str(i) for i in x)


def binary_columns(m):
    indices = np.arange(1 << m, dtype=np.int64)
    return [(indices >> (m-1-i)) & 1 for i in range(m)]


def all_energies(coeffs, columns, size):
    values = np.zeros(size, dtype=np.int64)
    for support, c in coeffs.items():
        term = c
        for i in support:
            term = term * columns[i]
        values += term
    return values


def product_series(models, smoke):
    summary, fibers, points, circuits = [], [], [], {}
    for model, spec in models.items():
        if smoke and model != "mixed":
            continue
        n, terms = spec["n"], spec["terms"]
        for scheme in ["shared", "separate"]:
            lift = product_lift(n,terms,scheme)
            m = len(lift.gates)
            ycols = binary_columns(m)
            circuits[f"{model}/{scheme}"] = dict(gates=lift.gates, readout=lift.readout,
                                                  loads=lift.loads, global_load=lift.global_load)
            configs = [(p,1) for p in ["directional","local","global"]]
            if (model,scheme) == ("mixed","shared"):
                configs.append(("directional",0))
            for penalty, delta in configs:
                coefs = lift.coefficients(penalty,delta)
                mags = [abs(c) for idx,c in coefs.items() if idx]
                base = dict(model=model,scheme=scheme,penalty=penalty,delta=delta)
                rowset = []
                for x in bits(n):
                    columns = list(x)+ycols
                    q = all_energies(coefs, columns, 1<<m)
                    f = poly_value(terms,x)
                    vio = np.zeros(1<<m, dtype=np.int64)
                    rhs = np.zeros_like(vio)
                    for g, (u,v) in enumerate(lift.gates):
                        parent = columns[u]*columns[v]
                        a,b = lift.parameters(penalty,delta,g)
                        p,nload = lift.loads[g]
                        rhs += (a-p)*(1-ycols[g])*parent + (b-nload)*ycols[g]*(1-parent)
                        vio += ycols[g] != parent
                    consistent = int(word(lift.consistent(x)),2)
                    slack = q - f - rhs
                    uniform_slack = q - f - delta*vio
                    row = dict(**base,x=word(x),f=f,min_q=int(q.min()),minimizers=int((q==q.min()).sum()),
                               consistent_y=word(lift.consistent(x)),consistent_q=int(q[consistent]),
                               min_slack=int(slack.min()),min_uniform_slack=int(uniform_slack.min()),min_invalid_excess=int((q-f)[vio>0].min()))
                    if row["min_q"] != f or q[consistent] != f or np.any(rhs < delta*vio) or slack.min() < 0 or (delta>0 and row["minimizers"]!=1):
                        raise AssertionError(row)
                    rowset.append(row)
                    if model == "mixed":
                        for j,(energy,v) in enumerate(zip(q,vio)):
                            points.append(dict(**base,x=word(x),y=format(j,f"0{m}b"),f=f,q=int(energy),violations=int(v),rhs16=int(rhs[j])))
                fibers.extend(rowset)
                summary.append(dict(**base,n=n,m=m,extensions=2**(n+m),minimum=min(r["f"] for r in rowset),
                                    tied_extensions=sum(r["minimizers"]-1 for r in rowset),min_abs=min(mags),
                                    max_abs=max(mags),coefficient_ratio=max(mags)/min(mags),
                                    min_invalid_excess=min(r["min_invalid_excess"] for r in rowset)))
    return summary, fibers, points, circuits


def logsumexp(arr, axis=None):
    maximum = arr.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(arr-maximum).sum(axis=axis, keepdims=True))).squeeze(axis=axis)


def marginal(f, q, beta):
    logfiber = logsumexp(-beta*(q-f[:,None]), axis=1)
    lp = -beta*f
    lq = lp + logfiber
    return np.exp(lp-logsumexp(lp)), np.exp(lq-logsumexp(lq))


def gibbs_series(smoke):
    rows, laws = [], []
    for m in ([2] if smoke else [1,2,3]):
        n = m+3
        terms = [(2, list(range(m+1))+[m+1]),(-3,list(range(m+1))+[m+2])]
        lift = product_lift(n, terms,"shared")
        assert len(lift.gates) == m
        ycols = binary_columns(m)
        ysums = sum(ycols)
        xs = list(bits(n))
        fchain = np.array([poly_value(terms,x) for x in xs],dtype=float)
        for delta in ([1] if smoke else [1,2]):
            coefs = lift.coefficients("directional",delta)
            qchain = np.array([all_energies(coefs,list(x)+ycols,1<<m) for x in xs],dtype=float)
            for beta in ([log(2)] if smoke else [log(2),1.,2.]):
                log_r = m*log1p(exp(-beta*delta))
                b1, b2 = tanh(log_r/4), m*exp(-beta*delta)/4
                a = -log_r/(2*beta)
                for M in [None,0,1,2,4,8]:
                    family = "ordinary" if M is None else "sharpness"
                    f = fchain if M is None else np.array([0.,a])
                    q = qchain if M is None else np.array([delta*ysums,a+(delta+M)*ysums])
                    p0, p1 = marginal(f,q,beta)
                    s0, s1 = marginal(f+19,q+19,beta)
                    tv = float(np.abs(p0-p1).sum()/2)
                    norm_err = max(abs(p0.sum()-1),abs(p1.sum()-1))
                    shift_err = max(np.max(abs(p0-s0)),np.max(abs(p1-s1)))
                    if norm_err>1e-12 or shift_err>1e-12 or tv>b1+1e-12 or b1>b2+1e-12 or (M==0 and tv>1e-12):
                        raise ValueError((family,m,beta,delta,M,tv))
                    key = dict(family=family,m=m,beta=beta,delta=delta,M="" if M is None else M)
                    rows.append(dict(**key,tv=tv,b1=b1,b2=b2,norm_error=float(norm_err),shift_error=float(shift_err)))
                    for x, p, marginal_p in zip(xs if M is None else [(0,),(1,)],p0,p1):
                        laws.append(dict(**key,x=word(x),p_target=float(p),p_lift=float(marginal_p)))
    return rows, laws
