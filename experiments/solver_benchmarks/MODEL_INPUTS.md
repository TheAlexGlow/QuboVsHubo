# Benchmark instances and objective conventions

The input manifest is `inputs/models.json`. It records the natural optimum, vertex or bit order, and transformation parameters for each case. Each Max-Cut objective maximizes the weight of an undirected cut; each HUBO objective minimizes the signed original polynomial. Every solver callback returns a loss to minimize. All costs are integer-valued.

| Case | Original bits | Transformed bits | Exact natural optimum | Source |
|---|---:|---:|---:|---|
| triangle | 3 | 3 | 5 | B2 weighted triangle |
| tree | 6 | 6 | 5 | B2 five-edge unit tree |
| g05_60.0 | 60 | 60 | 536 | Biq Mac G0.5 |
| g05_60.1 | 60 | 60 | 532 | Biq Mac G0.5 |
| mixed | 5 | 7 | −3 | B2 shared signed control |
| f4 | 7 | 16 | −7 | B2 polynomial (49) |
| coupled_mixed | 15 | 21 | −9 | Three disjoint mixed gadgets and two nonnegative links |

## Data acquisition

The public Biq Mac Library provides the individual benchmark files. The source graph format starts with vertex and edge counts on the first line, followed by one undirected edge (one-based endpoints and weight) on each subsequent line. Both graphs have 885 distinct edges of weight one. A. Wiegele, *Biq Mac Library – A collection of Max-Cut and quadratic 0-1 programming instances of medium size*, Table 6, p. 9, gives the two optimal cut values. [Format and family](https://biqmac.aau.at/biqmaclib.html) · [Optimal values](https://biqmac.aau.at/biqmaclib.pdf)

The repository stores the download URLs and SHA-256 digests. Download and verify the original bytes with the command below. Set `QUBO_GRAPH_DIR` to use the same directory in subsequent runs; the default location is the user's cache under qubo-vs-hubo/biqmac. Existing correctly hashed files work offline. Upstream's catalogue provides the files for testing and does not identify a separate redistribution license.

```sh
export QUBO_GRAPH_DIR=/tmp/qubo-b3-public-graphs
python fetch_graphs.py --graph-dir "$QUBO_GRAPH_DIR"
```

Graph identifiers refer to independent fixed instances from the same G0.5 family. No optimizer outcomes were consulted when choosing them. The first public pivot is vertex zero in zero-based indexing. Its lowest-numbered adjacent vertex supplies the second pivot, yielding ordered pairs (0,1) and (0,2). Both pivots have other neighbors. The triangle and tree use their preset B2 pairs (0,1) and (1,0). The stored order is zero through n−1, with all auxiliary bits appended in topological order in lifts. The XOR map sets x[u] equal to z[u] xor z[v] and fixes other bits. The solver evaluates the polynomial comprising the surviving quadratic edges, the special uv term and cubic parities for all remaining edges incident on u. The decoded natural score uses a direct weighted cut calculation.

Each HUBO uses the same shared sorted-prefix circuit in both quadratic formulations. Directional weights are A=P+1 and B=N+1; local symmetric weights are A=B=max(P,N)+1. The native HUBO score is the original unlifted energy. The joint solver loss includes the quadratic penalties on the actual returned auxiliary bits. The coupled instance is the sum of three copies of the signed mixed objective on consecutive five-bit blocks, plus (x0−x5)²+2(x5−x10)². The gadgets each attain −3 at (1,1,1,0,1), and these choices make the added nonnegative terms zero. Its optimum is therefore −9. Full enumeration of 32,768 original states gives the same value.

## Reuse and pointwise API

The copies of `exact_api.py` and `inputs/controls-v1.json` are byte-for-byte from B2 code commit `dc450e1180dffe127246726d14611375d76ace11`. The manifest records SHA-256 digests of those files and the complete upstream B2 input table. B2 results commit `792b3d1259749913f11c72385f75c6aa403527ef` records the exact enumerations. This checkout is self-contained.

```python
from models import CASE_IDS, formulations, load
model = load("mixed")
losses = model.evaluate("directional", [[1,1,1,0,1,1,1]])
original, score, feasible = model.decode("directional", [1,1,1,0,1,1,1])
```

`CASE_IDS` and `formulations(case)` preserve the frozen case and formulation order. Graph labels are quadratic/xor; HUBO labels are original/directional/local. Calls `load(case,graph_dir=None)`, `model.dimension(form)`, `model.evaluate(form,batch)` and `model.decode(form,bits)` provide model loading, transformed dimension, vectorized solver loss, and candidate decoding. Scores and objective losses use the original, unnormalized integer units. Model attributes n, optimum, maximize, case and order are defined on both classes. Graphs expose edges, u and v. The HUBO exposes terms, its two collected Q dictionaries in coefficients, delta, and the lift object with its topologically ordered gates, signed loads and readout.
