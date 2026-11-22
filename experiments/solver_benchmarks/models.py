"""Self-contained equivalent Max-Cut and product-lift models for the benchmark."""
import numpy as np
from exact_api import cut_score, decode_xor, poly_value, product_lift
from fetch_graphs import SPECS, fetch_graph

CASE_IDS = tuple(SPECS)
GRAPH_FORMS = ("quadratic","xor")
HUBO_FORMS = ("original","directional","local")


def formulations(case):
    return GRAPH_FORMS if SPECS[case]["family"]=="maxcut" else HUBO_FORMS


def checked_batch(points,dimension):
    batch = np.asarray(points)
    if batch.ndim!=2 or batch.shape[1]!=dimension:
        raise ValueError(f"Expected an array with shape (batch,{dimension}), got {batch.shape}")
    if np.any((batch!=0) & (batch!=1)):
        raise ValueError("All assignments must consist exclusively of binary labels")
    return batch.astype(np.int64,copy=False)


def polynomial_batch(terms,points):
    values = np.zeros(len(points),dtype=np.int64)
    for coefficient,support in terms:
        v = coefficient
        for i in support:
            v = v * points[:,i]
        values += v
    return values


def edge_rows(case,spec,graph_dir):
    if "edges" in spec:
        return [tuple(edge) for edge in spec["edges"]]
    rows = [tuple(map(int,line.split())) for line in fetch_graph(case,graph_dir).splitlines() if line.strip()]
    n,m = rows[0]
    edges = [(i-1,j-1,w) for i,j,w in rows[1:]]
    if n!=spec["n"] or m!=spec["n_edges"] or len(edges)!=m:
        raise ValueError(f"Invalid graph dimensions in {case}")
    pairs = {tuple(sorted((i,j))) for i,j,w in edges}
    if len(pairs)!=m or any(i<0 or i>=j or j>=n for i,j in pairs):
        raise ValueError(f"Duplicate edges, loops or invalid vertices in {case}")
    return edges


def edge_arrays(edges):
    return np.asarray(edges,dtype=np.int64).reshape(-1,3).T


def quadratic_edges(batch,arrays):
    left,right,w = arrays
    a,b = batch[:,left],batch[:,right]
    return (a+b-2*a*b) @ w


class MaxCut:
    maximize = True

    def __init__(self,case,spec,graph_dir):
        self.case = case
        self.n = spec["n"]
        self.optimum = spec["optimum"]
        self.order = tuple(spec["order"])
        self.u, self.v = spec["u"],spec["v"]
        self.edges = edge_rows(case,spec,graph_dir)
        self._all = edge_arrays(self.edges)
        self._fixed = edge_arrays([e for e in self.edges if self.u not in e[:2]])
        self._uv_weight = 0
        changed = []
        for i,j,w in self.edges:
            if self.u in (i,j):
                k = i if j==self.u else j
                if k==self.v:
                    self._uv_weight = w
                else:
                    changed.append((k,w))
        if not changed:
            raise ValueError("A genuine cubic requires a second neighbor at the XOR pivot")
        self._k,self._w = np.asarray(changed,dtype=np.int64).T

    def dimension(self,form):
        if form not in GRAPH_FORMS:
            raise ValueError(form)
        return self.n

    def evaluate(self,form,batch):
        z = checked_batch(batch,self.dimension(form))
        if form=="quadratic":
            return -quadratic_edges(z,self._all)
        a,b,c = z[:,self.u:self.u+1],z[:,self.v:self.v+1],z[:,self._k]
        parity3 = a+b+c - 2*(a*b+a*c+b*c) + 4*a*b*c
        score = quadratic_edges(z,self._fixed) + self._uv_weight*z[:,self.u] + parity3@self._w
        return -score

    def decode(self,form,bits):
        z = checked_batch(np.asarray(bits)[None,:],self.dimension(form))[0]
        x = np.array(decode_xor(z,self.u,self.v) if form=="xor" else z, dtype=np.int64)
        return x,int(cut_score(self.edges,x)),True


class Hubo:
    maximize = False
    delta = 1

    def __init__(self,case,spec):
        self.case = case
        self.n = spec["n"]
        self.order = tuple(spec["order"])
        self.optimum = spec["optimum"]
        self.terms = [(c,tuple(s)) for c,s in spec["terms"]]
        self.lift = product_lift(self.n,self.terms,"shared")
        self.coefficients = {p:self.lift.coefficients(p,self.delta) for p in ("directional","local")}
        self._polynomials = {p:[(c,s) for s,c in coefs.items()] for p,coefs in self.coefficients.items()}
        self._polynomials["original"] = self.terms

    def dimension(self,form):
        if form not in HUBO_FORMS:
            raise ValueError(form)
        return self.n if form=="original" else self.n+len(self.lift.gates)

    def evaluate(self,form,batch):
        z = checked_batch(batch,self.dimension(form))
        return polynomial_batch(self._polynomials[form],z)

    def decode(self,form,bits):
        z = checked_batch(np.asarray(bits)[None,:],self.dimension(form))[0]
        x = z[:self.n].copy()
        feasible = form=="original" or self.lift.consistent(x)==tuple(z[self.n:])
        return x,int(poly_value(self.terms,x)),bool(feasible)


def load(case,graph_dir=None):
    spec = SPECS[case]
    if spec["family"]=="maxcut":
        return MaxCut(case,spec,graph_dir)
    return Hubo(case,spec)
