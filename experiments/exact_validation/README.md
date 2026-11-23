# Exact validation of product lifts, Gibbs marginals and unfolding ranks

From the repository root, change to `experiments/exact_validation`. Use a CPython 3.9 environment with `python -m pip install -r requirements.lock`. All commands below run in that directory. For a temporary checkout or a comparison against the recorded outputs, choose a fresh output directory via `--out`.

```sh
python run.py smoke --out /tmp/qubo-exact-smoke
python run.py full --out results
python report.py --results results --out results/derived
```

The first command executes the mixed signed circuit, finite-temperature controls, triangle and copy diagnostic. The second runs the protocol in PROTOCOL.md and writes raw and derived output. Each run checks exact identities and bounds as it enumerates. The third reads the raw CSV independently of all enumeration code. The independent audit is in check_independent.py. The measured elapsed time and machine metadata in run.json can vary on replay. The code revision defaults to the last commit changing this experiment's inputs, implementation or instructions.

## Small-instance interface, version 1

`exact_api.py` has no third-party dependencies. Import `cut_score(edges,x)`, `decode_xor(z,u,v)` and `cubic_cut(edges,z,u,v)` for the unnormalized score to maximize. Each edge is (i,j,weight), listed once, vertex indices start at zero. For a polynomial specified by (signed integer coefficient, list of distinct zero-based indices), `product_lift(n,terms,scheme)` yields the topologically ordered gate input pairs, readout, ancestor loads, a consistent extension, and a coefficient dictionary for any specified penalty choice. `check_fiber(n,terms,x,scheme,penalty,delta)` enumerates all auxiliaries and returns the minimum and number of minimizers, checking the pointwise identity and inequality. Lift energies are minimized. Positive margins require a unique minimizer at fixed x. Integer regression fixtures, including the invalid star substitution, are in inputs/controls-v1.json. The numeric series checks this file against the executable API first.

## Files and column conventions

Blank cells denote inapplicable quantities. Assignment words are most significant bit first, with original bits before auxiliaries. Integer ranks denote rank over the reals except explicitly binary kappa. Every Gibbs temperature is in reciprocal objective-energy units. Each row belongs to the run_id in the containing directory's run.json.

| Raw file | Row and columns |
|---|---|
| products.csv | One model, prefix scheme, penalty and margin. n and m count original bits and gates. extensions counts all assignments. minimum is the original minimum. tied_extensions counts inconsistent minimizers. min_invalid_excess is the smallest Q−f over inconsistent assignments. min_abs, max_abs and coefficient_ratio exclude constant and zero coefficients. |
| fibers.csv | One row per original x and product configuration. f, min_q, minimizers, consistent_y and consistent_q describe the fiber. min_slack is the minimum over y of Q−f minus the full (16) right-hand side, using A−P for each false negative and B−N for each false positive. min_uniform_slack uses the uniform delta per violation. |
| mixed_points.csv | Every mixed-control x,y with original energy f, lifted q with the number of violated actual gates and the weighted right-hand side rhs16. |
| circuits.json | Input pairs for each gate, quadratic readout, [P,N] by gate, and global load before margin. Gate g has absolute variable index n+g. |
| gibbs.csv | family, m, beta, delta, interaction M for the sharpness family, TV, both (26) bounds b1,b2, maximum probability normalization and shift errors. |
| probabilities.csv | Target and marginalized lift probabilities at each original state for every Gibbs row. |
| ranks.csv | One tensor, coordinate map, fixed order, and cut after k variables. exact_rank is computed by integer elimination; eta_0p1 and eta_0p01 are Frobenius-tail ranks. The original_* columns are the identity-coordinate comparators at the identical cut. Objective rows give degree, coefficient_rank, support_matching on actual support, crossing_terms, cross_quadratic_rank when degree ≤2, and degree_cap. kappa and inverse_kappa use binary rank; split_pairs counts copy pairs crossing the cut. lift_energy is the full quadratic copy energy, original_energy the zero target, joint_gibbs the full joint, and marginal_gibbs its original-bit marginal. |
| coefficient_draws.json | Realized integer coefficients in each independent-coefficient control, including the tied four-term comparator. |
| degree_checks.csv | For each (left,right,p), the maximum matching size in the full degree-p-or-lower support and the formula (41) value. |
| run.json | Revision, input hashes, version, seed and thread settings, row counts, platform and elapsed seconds. |

The generated penalty table compares schemes at identical gap. The Gibbs table and chart use the matched m=2, delta=1 slice. Their finite-sharpness panel fixes beta=ln(2). The coordinate chart compares representations of a fixed tree in natural order (triangle in smoke mode). The complete per-instance and per-order screening table states the selected map and its measured metrics under the preregistered ordering. For copy Gibbs with t=1/2, each split pair supplies one invertible two-by-two matrix, while each unsplit pair contributes a nonzero vector on one side. This proves exact rank 2 to the power split_pairs. For the tree edge encoding, multiplying the local parity transitions from the root decodes every vertex uniquely, and the cut score is the weighted sum of the child edge bits.
