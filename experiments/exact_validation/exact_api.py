"""Version 1. Standard-library pointwise Max-Cut and product-lift reference API."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from math import prod

API_VERSION = 1


def bits(n):
    return product((0, 1), repeat=n)


def poly_value(terms, x):
    return sum(c * prod(x[i] for i in factors) for c, factors in terms)


def cut_score(edges, x):
    return sum(w * (x[i] ^ x[j]) for i, j, w in edges)


def decode_xor(z, u, v):
    x = list(z)
    x[u] ^= x[v]
    return tuple(x)


def cubic_cut(edges, z, u, v):
    """Equation (37), including the special edge uv. Return the score to maximize."""
    score = 0
    for i, j, weight in edges:
        if u not in (i, j):
            a, b = z[i], z[j]
            parity = a + b - 2*a*b
        else:
            k = j if i == u else i
            if k == v:
                parity = z[u]
            else:
                a, b, c = z[u], z[v], z[k]
                parity = a+b+c - 2*(a*b + a*c + b*c) + 4*a*b*c
        score += weight * parity
    return score


@dataclass
class Lift:
    n: int
    gates: list
    readout: list
    loads: list
    global_load: int

    def coefficients(self, penalty, delta):
        """Collect all multilinear Q coefficients; keys are sorted variable tuples."""
        coefficients = defaultdict(int)

        def add(c, *indices):
            coefficients[tuple(sorted(set(indices)))] += c

        for c, indices in self.readout:
            add(c, *indices)
        for g, (u, v) in enumerate(self.gates):
            a,b = self.parameters(penalty,delta,g)
            y = self.n + g
            add(a, u, v)
            add(-(a+b), u, y)
            add(-(a+b), v, y)
            add(a+2*b, y)
        return {s:c for s,c in coefficients.items() if c}

    def parameters(self,penalty,delta,g):
        p,nload = self.loads[g]
        if penalty == "directional":
            return p+delta,nload+delta
        if penalty == "local":
            weight = max(p,nload)+delta
        elif penalty == "global":
            weight = self.global_load+delta
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
        return weight,weight

    def consistent(self, x):
        z = list(x)
        for u, v in self.gates:
            z.append(z[u]*z[v])
        return tuple(z[self.n:])

    def violations(self, x, y):
        z = tuple(x) + tuple(y)
        return sum(y[g] != z[u]*z[v] for g, (u,v) in enumerate(self.gates))


def product_lift(n, terms, scheme):
    """Create shared or separate sorted-prefix chains in input term order."""
    if scheme not in ("shared", "separate"):
        raise ValueError("scheme must be shared or separate")
    gates, readout, loads = [], [], []
    cache = {}
    global_load = 0
    for c, support in terms:
        factors = sorted(support)
        if len(set(factors)) != len(factors) or any(i < 0 or i >= n for i in factors):
            raise ValueError("supports must be multilinear on the original variables")
        ancestors = []
        while len(factors) > 2:
            u, v = factors[:2]
            pair = tuple(sorted((u,v)))
            if scheme == "shared" and pair in cache:
                g = cache[pair]
            else:
                g = len(gates)
                cache[pair] = g
                gates.append((u,v))
                loads.append([0,0])
            ancestors.append(g)
            factors = [n+g] + factors[2:]
        readout.append((c, tuple(factors)))
        if ancestors:
            global_load += abs(c)
        for g in set(ancestors):
            loads[g][0] += max(0,c)
            loads[g][1] += max(0,-c)
    return Lift(n, gates, readout, loads, global_load)


def check_fiber(n, terms, x, scheme="shared", penalty="directional", delta=1):
    """Enumerate one small integer fiber. Return its pointwise minimum and multiplicity."""
    lift = product_lift(n, terms, scheme)
    items = [(c,s) for s,c in lift.coefficients(penalty,delta).items()]
    f = poly_value(terms, x)
    energies = []
    for y in bits(len(lift.gates)):
        q = poly_value(items, tuple(x)+y)
        z = tuple(x)+y
        rhs = 0
        for g,(u,v) in enumerate(lift.gates):
            a,b = lift.parameters(penalty,delta,g)
            p,nload = lift.loads[g]
            parent = z[u]*z[v]
            rhs += (a-p)*(1-y[g])*parent + (b-nload)*y[g]*(1-parent)
        if q-f < rhs:
            raise ValueError("violation-bound failure")
        energies.append(q)
    minimum = min(energies)
    if minimum != f or (delta > 0 and energies.count(minimum) != 1):
        raise ValueError("pointwise exactness or uniqueness failure")
    return minimum, energies.count(minimum)
