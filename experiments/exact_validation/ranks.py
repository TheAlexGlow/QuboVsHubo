"""Integer unfolding ranks, structural bounds, and relative Frobenius tails."""
from itertools import combinations
from math import comb, gcd, log
import random
import numpy as np

from exact_api import bits, cut_score, cubic_cut, decode_xor, poly_value


def integer_rank(matrix):
    """Integer row additions and nonzero scalings preserve rank over Q and R."""
    a = [list(map(int,row)) for row in matrix]
    rank = 0
    for j in range(len(a[0])):
        pivot = next((i for i in range(rank,len(a)) if a[i][j]),None)
        if pivot is None:
            continue
        a[rank],a[pivot] = a[pivot],a[rank]
        for i in range(rank+1,len(a)):
            b,p = a[i][j],a[rank][j]
            if b:
                g = gcd(b,p)
                row = [(p//g)*v - (b//g)*w for v,w in zip(a[i],a[rank])]
                d = 0
                for v in row:
                    d = gcd(d,v)
                a[i] = [v//d for v in row] if d>1 else row
        rank += 1
        if rank==len(a):
            break
    return rank


def gf2_rank(matrix):
    basis = {}
    for row in matrix:
        v = 0
        for i in row:
            v = (v<<1) | int(i)
        while v:
            pivot = v.bit_length()-1
            if pivot not in basis:
                basis[pivot] = v
                break
            v ^= basis[pivot]
    return len(basis)


def kappa(mat, order, k):
    left,right = order[:k],order[k:]
    return gf2_rank(mat[np.ix_(left,right)]) + gf2_rank(mat[np.ix_(right,left)])


def matching(supports):
    adjacency = {}
    for i,j in supports:
        adjacency.setdefault(i,[]).append(j)
    right_match = {}
    def augment(i,seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in right_match or augment(right_match[j],seen):
                right_match[j] = i
                return True
        return False
    return sum(augment(i,set()) for i in adjacency)


def pop(i):
    return bin(i).count("1")


def rank_cap(k,l,p):
    def partial(d,q):
        return sum(comb(d,i) for i in range(min(d,q)+1))
    return min(partial(k,q)+partial(l,p-q-1) for q in range(-1,p+1))


def degree_checks(smoke):
    rows = []
    for k in ([2] if smoke else range(1,5)):
        for l in ([3] if smoke else range(1,5)):
            for p in range(k+l+1):
                match = matching((i,j) for i in range(1<<k) for j in range(1<<l) if pop(i)+pop(j)<=p)
                cap = rank_cap(k,l,p)
                assert cap==match
                rows.append(dict(left=k,right=l,p=p,matching=match,degree_cap=cap))
    return rows


def mobius(values,n):
    c = list(values)
    for b in range(n):
        for mask in range(1<<n):
            if mask & (1<<b):
                c[mask] -= c[mask^(1<<b)]
    return c


def zeta(d):
    return np.array([[int(i&j == j) for j in range(1<<d)] for i in range(1<<d)],dtype=np.int64)


def approx(matrix):
    f = np.asarray(matrix,dtype=float)
    if np.max(abs(f))==0:
        return 0,0
    s = np.linalg.svd(f/np.max(abs(f)),compute_uv=False)
    tail = np.r_[np.cumsum((s[::-1]**2))[::-1],0.]
    return tuple(next(i for i,t in enumerate(tail) if t <= eta**2*tail[0]*(1+1e-12)) for eta in (.1,.01))


def tree_maps(parents):
    def decode(z):
        x = list(z)
        for i,p in enumerate(parents):
            if p>=0:
                x[i] ^= x[p]
        return tuple(x)

    def encode(x):
        return tuple(x[i] if p<0 else x[i]^x[p] for i,p in enumerate(parents))
    return decode,encode


def linear_matrix(transform,n):
    basis = np.eye(n,dtype=int)
    return np.array([transform(e) for e in basis]).T


def unfold(values,n,order,k):
    return np.array(values,dtype=object).reshape([2]*n).transpose(order).reshape((1<<k,-1))


def approx_row(key,values,analytic=None):
    exact = integer_rank(values)
    if analytic is not None:
        assert exact == analytic,(key,exact,analytic)
    r1,r2 = approx(values)
    assert r1<=r2<=exact
    return dict(**key,exact_rank=exact,eta_0p1=r1,eta_0p01=r2)


def analyze_objective(name,n,score,orders,transforms,edges=None,u=None,v=None):
    assignments = list(bits(n))
    rows = []
    baseline = {}
    for label,(forward,inverse) in transforms.items():
        mat, inv = linear_matrix(forward,n),linear_matrix(inverse,n)
        assert np.array_equal((mat@inv)%2,np.eye(n,dtype=int))
        values = []
        for z in assignments:
            x = forward(z)
            assert inverse(x)==z and forward(inverse(z))==z
            value = score(x)
            if edges is not None and label=="xor":
                assert cubic_cut(edges,z,u,v)==value
            values.append(int(value))
        coeffs = mobius(values,n)
        degree = max(pop(i) for i,c in enumerate(coeffs) if c)
        # At beta=log(2), exactly integer-proportional to the normalized Gibbs tensor.
        min_score = min(values)
        weights = [1 << (f-min_score) for f in values]
        for order_name,order in orders.items():
            for k in range(1,n):
                l = n-k
                key = dict(instance=name,map=label,order=order_name,order_labels=",".join(map(str,order)),
                           cut=k,kappa=kappa(mat,order,k),inverse_kappa=kappa(inv,order,k))
                f = unfold(values,n,order,k)
                c = unfold(coeffs,n,order,k)
                assert np.array_equal(zeta(k)@c@zeta(l).T, f)
                support = [(i,j) for i in range(1<<k) for j in range(1<<l) if c[i,j]]
                cross = sum(i!=0 and j!=0 for i,j in support)
                match = matching(support)
                r = approx_row(dict(**key,tensor="objective"),f)
                c_rank = integer_rank(c)
                cap = rank_cap(k,l,degree)
                assert r["exact_rank"]==c_rank<=min(match,cross+2,cap,1<<min(k,l))
                cross_rank = ""
                if degree<=2:
                    cross_rank = integer_rank([[c[1<<i,1<<j] for j in range(l)] for i in range(k)])
                    assert c_rank<=cross_rank+2
                r.update(degree=degree,coefficient_rank=c_rank,support_matching=match,crossing_terms=cross,
                         cross_quadratic_rank=cross_rank,degree_cap=cap)
                joint = approx_row(dict(**key,tensor="gibbs"),unfold(weights,n,order,k))
                for entry in (r,joint):
                    bkey = (order_name,k,entry["tensor"])
                    metrics = ("exact_rank","eta_0p1","eta_0p01")
                    current = tuple(entry[v] for v in metrics)
                    if label=="identity":
                        baseline[bkey] = current
                    for new,old in zip(current,baseline[bkey]):
                        assert new <= (1<<key["kappa"])*old and old <= (1<<key["inverse_kappa"])*new
                    entry["original_exact_rank"],entry["original_eta_0p1"],entry["original_eta_0p01"] = baseline[bkey]
                    rows.append(entry)
    return rows


def copy_ranks(smoke):
    rows = []
    for m in ([2] if smoke else [2,3,4]):
        pairs = [(i,m+i) for i in range(m)]
        orders = {"grouped":list(range(2*m)), "interleaved":[j for pair in pairs for j in pair]}
        energy = [sum(z[i]^z[j] for i,j in pairs) for z in bits(2*m)]
        weights = [1<<(m-e) for e in energy]
        # Exact integer marginal fiber weights; all original assignments have equal weight.
        marg = np.array(weights,dtype=object).reshape(1<<m,-1).sum(axis=1)
        assert len(set(marg))==1
        for label,order in orders.items():
            for k in range(1,2*m):
                split = sum((i in order[:k]) != (j in order[:k]) for i,j in pairs)
                key = dict(instance=f"copy{m}",map="identity",order=label,order_labels=",".join(map(str,order)),
                           cut=k,split_pairs=split)
                for tensor,v,expected in [("lift_energy",energy,None),("joint_gibbs",weights,1<<split)]:
                    rows.append(approx_row(dict(**key,tensor=tensor),unfold(v,2*m,order,k),expected))
        for k in range(1,m):
            key = dict(instance=f"copy{m}",map="identity",order="original",order_labels=",".join(map(str,range(m))),cut=k)
            rows.append(approx_row(dict(**key,tensor="marginal_gibbs"),marg.reshape(1<<k,-1),1))
            rows.append(approx_row(dict(**key,tensor="original_energy"),np.zeros((1<<k,1<<(m-k)),dtype=int),0))
    return rows


def fixed_generic(support,n,seed):
    def mask(s):
        return sum(1<<(n-1-i) for i in s)
    rng = random.Random(seed)
    available = list(range(-9,0))+list(range(1,10))
    return [(rng.choice(available),s) for s in sorted(support,key=mask)]


def rank_series(inputs,smoke):
    rows = []
    generated = {}
    identity = lambda z: tuple(z)
    for name,spec in inputs["graphs"].items():
        if smoke and name != "triangle":
            continue
        n,edges,u,v = (spec[key] for key in ("n","edges","u","v"))
        xor = lambda z: decode_xor(z,u,v)
        maps = {"identity":(identity,identity),"xor":(xor,xor)}
        if "parents" in spec:
            maps["tree_edge"] = tree_maps(spec["parents"])
        rows.extend(analyze_objective(name,n,lambda x:cut_score(edges,x),spec["orders"],maps,edges,u,v))
        if name=="tree":
            support = [(i,) for i in range(n)] + [(i,j) for i,j,w in edges]
            for seed in inputs["free_draw_seeds"]:
                instance = f"tree_free_{seed}"
                terms = fixed_generic(support,n,seed)
                generated[instance] = terms
                rows.extend(analyze_objective(instance,n,lambda x:poly_value(terms,x),spec["orders"],maps))
    if not smoke:
        support = list((i,j) for i in (0,1) for j in (2,3))
        maps = {"identity":(identity,identity)}
        for seed in [None]+inputs["free_draw_seeds"]:
            name = "cross_tied" if seed is None else f"cross_free_{seed}"
            terms = [(1,s) for s in support] if seed is None else fixed_generic(support,4,seed)
            generated[name] = terms
            rows.extend(analyze_objective(name,4,lambda x:poly_value(terms,x),inputs["graphs"]["square"]["orders"],maps))
    rows.extend(copy_ranks(smoke))
    return rows,generated
