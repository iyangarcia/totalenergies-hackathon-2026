"""
Exact-ish solver for the transformer placement problem.

The problem is a k-median / uncapacitated facility location problem with an
extra "double coverage" constraint for industries:

    choose exactly K spots (facilities) out of the valid candidate spots to
    minimize the total Manhattan distance from every hospital/industry
    (demand point) to its nearest chosen spot, subject to every industry
    having at least 2 chosen spots within Manhattan distance 3.

Two pieces:
  - greedy_construct(): a fast, deterministic, vectorized (numpy) greedy
    heuristic. Used standalone as a fallback and as a warm-start hint for
    the ILP below.
  - solve_ilp(): an exact formulation solved with Google OR-Tools CP-SAT.
    Given a time limit it returns the best solution found, which is
    provably optimal if the solver reports OPTIMAL before the limit.
"""
import os
import time

import numpy as np
from ortools.sat.python import cp_model

from grid import manhattan_matrix, score_configuration


def greedy_construct(grid, dist_demand, dist_industry, k, radius=3, required=2):
    """Deterministic greedy: first satisfy industry double-coverage with the
    fewest/most useful spots, then fill remaining slots by greedy facility
    location (each pick minimizes the resulting total demand distance)."""
    num_spots = len(grid.valid_spots)
    num_industries = len(grid.industries)
    chosen = []
    chosen_mask = np.zeros(num_spots, dtype=bool)

    if num_industries > 0:
        remaining_need = np.full(num_industries, required, dtype=np.int32)
        covers = dist_industry <= radius  # (num_industries, num_spots) bool

        while remaining_need.sum() > 0 and len(chosen) < k:
            active_rows = remaining_need > 0
            gains = covers[active_rows][:, ~chosen_mask].sum(axis=0)
            if gains.size == 0 or gains.max() == 0:
                break  # no candidate can help any remaining industry
            candidate_positions = np.where(~chosen_mask)[0]
            best_local = np.argmax(gains)
            best_spot = candidate_positions[best_local]

            chosen.append(int(best_spot))
            chosen_mask[best_spot] = True
            covered_now = covers[:, best_spot] & active_rows
            remaining_need[covered_now] -= 1
            remaining_need = np.clip(remaining_need, 0, None)

    # Greedy facility-location fill for the rest of the budget.
    if chosen:
        current_min = dist_demand[:, chosen].min(axis=1)
    else:
        current_min = np.full(dist_demand.shape[0], np.iinfo(np.int32).max, dtype=np.int64)

    while len(chosen) < k:
        remaining_idx = np.where(~chosen_mask)[0]
        if remaining_idx.size == 0:
            break
        candidate_cols = dist_demand[:, remaining_idx]
        combined = np.minimum(current_min[:, None], candidate_cols)
        totals = combined.sum(axis=0)
        best_local = int(np.argmin(totals))
        best_spot = int(remaining_idx[best_local])

        chosen.append(best_spot)
        chosen_mask[best_spot] = True
        current_min = combined[:, best_local]

    return chosen


def _candidate_lists(dist_demand, candidates_per_demand):
    """For each demand point, indices of its `candidates_per_demand` closest
    spots, or every spot if the pool is already small enough."""
    num_demand, num_spots = dist_demand.shape
    if num_spots <= candidates_per_demand:
        full = list(range(num_spots))
        return [full for _ in range(num_demand)]
    order = np.argsort(dist_demand, axis=1)[:, :candidates_per_demand]
    return [row.tolist() for row in order]


def solve_ilp(grid, dist_demand, dist_industry, k, time_limit_s=10.0,
              candidates_per_demand=60, radius=3, required=2, num_workers=None):
    num_spots = len(grid.valid_spots)
    num_demand = dist_demand.shape[0]
    num_industries = dist_industry.shape[0]

    if num_spots < k:
        raise ValueError(f"Only {num_spots} valid spots but {k} transformers requested.")

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{s}") for s in range(num_spots)]
    model.Add(sum(x) == k)

    for t in range(num_industries):
        near = np.where(dist_industry[t] <= radius)[0]
        if len(near) < required:
            raise ValueError(f"Industry #{t} has only {len(near)} valid spots within radius {radius}; "
                              f"needs {required}. This map cannot satisfy the coverage rule.")
        model.Add(sum(x[s] for s in near) >= required)

    candidates = _candidate_lists(dist_demand, candidates_per_demand)
    y = {}
    cost_terms = []
    for d in range(num_demand):
        row_vars = []
        for s in candidates[d]:
            var = model.NewBoolVar(f"y{d}_{s}")
            y[(d, s)] = var
            row_vars.append(var)
            model.Add(var <= x[s])
            cost_terms.append(int(dist_demand[d, s]) * var)
        model.Add(sum(row_vars) == 1)

    model.Minimize(sum(cost_terms))

    # Warm start: hint x AND a consistent y assignment, so CP-SAT can accept the
    # greedy solution as an immediate feasible incumbent instead of spending the
    # whole time budget on presolve/search before finding its first solution.
    warm_start = greedy_construct(grid, dist_demand, dist_industry, k, radius, required)
    warm_set = set(warm_start)
    greedy_score = score_configuration(dist_demand, warm_start)
    for s in range(num_spots):
        model.add_hint(x[s], 1 if s in warm_set else 0)
    for d in range(num_demand):
        in_warm = [s for s in candidates[d] if s in warm_set]
        if not in_warm:
            continue
        best_s = min(in_warm, key=lambda s: dist_demand[d, s])
        for s in candidates[d]:
            model.add_hint(y[(d, s)], 1 if s == best_s else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers or max(1, os.cpu_count() or 1)

    t0 = time.perf_counter()
    status = solver.Solve(model)
    wall_time = time.perf_counter() - t0

    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # CP-SAT ran out of time before finding any feasible solution on a large
        # instance -- fall back to the greedy construction, which is always valid.
        exact_score = greedy_score
        return {
            "chosen": warm_start,
            "status": "GREEDY_FALLBACK",
            "objective_reported": greedy_score,
            "best_bound": None,
            "exact_score": exact_score,
            "proven_optimal": False,
            "wall_time": wall_time,
        }

    chosen = [s for s in range(num_spots) if solver.Value(x[s]) == 1]
    objective = int(solver.ObjectiveValue())
    best_bound = int(solver.BestObjectiveBound())
    exact_score = score_configuration(dist_demand, chosen)
    if exact_score > greedy_score:
        # Pruned candidate lists can in rare cases make the ILP's own solution
        # look worse than greedy for the y-restricted objective; never regress.
        chosen, exact_score, objective = warm_start, greedy_score, greedy_score

    return {
        "chosen": chosen,
        "status": status_name,
        "objective_reported": objective,
        "best_bound": best_bound,
        "exact_score": exact_score,
        "proven_optimal": status == cp_model.OPTIMAL,
        "wall_time": wall_time,
    }
