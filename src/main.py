"""
Transformer placement solver.

Formulates the problem exactly (facility location + industry double-coverage,
Manhattan distance since the whole grid is traversable) and solves it with
Google OR-Tools CP-SAT, seeded with a fast greedy warm start. No LLM in the
loop: choosing where to place transformers is a deterministic optimization
problem, not a language task.
"""
import argparse
import os
import time

from grid import Grid, manhattan_matrix, score_configuration, industry_coverage_ok
from solver import greedy_construct, solve_ilp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "inputs")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="1.txt", help="Map filename inside inputs/, or a full path.")
    p.add_argument("--count", type=int, default=28, help="Number of transformers ('C') to place.")
    p.add_argument("--time-limit", type=float, default=10.0, help="CP-SAT time budget in seconds.")
    p.add_argument("--candidates", type=int, default=60,
                   help="Max nearest-spot candidates considered per demand point (keeps large maps tractable).")
    p.add_argument("--greedy-only", action="store_true",
                   help="Skip the ILP and just report the greedy warm-start solution.")
    p.add_argument("--output", default=None, help="Output path (default: outputs/<input filename>).")
    return p.parse_args()


def resolve_input_path(name):
    if os.path.isabs(name) or os.path.exists(name):
        return name
    return os.path.join(DEFAULT_INPUT_DIR, name)


def main():
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, os.path.basename(input_path))

    grid = Grid(input_path)
    print(f"Map: {grid.rows}x{grid.cols} | valid spots: {len(grid.valid_spots)} | "
          f"hospitals: {len(grid.hospitals)} | industries: {len(grid.industries)}")

    if len(grid.valid_spots) < args.count:
        raise SystemExit(f"ERROR: only {len(grid.valid_spots)} valid spots but {args.count} transformers requested.")

    dist_demand = manhattan_matrix(grid.demand_points, grid.valid_spots)
    dist_industry = manhattan_matrix(grid.industries, grid.valid_spots)

    t0 = time.perf_counter()
    if args.greedy_only:
        chosen_idx = greedy_construct(grid, dist_demand, dist_industry, args.count)
        elapsed = time.perf_counter() - t0
        score = score_configuration(dist_demand, chosen_idx)
        covered = industry_coverage_ok(dist_industry, chosen_idx)
        print(f"[greedy] score={score}  industry-coverage-ok={covered}  time={elapsed:.4f}s")
    else:
        result = solve_ilp(grid, dist_demand, dist_industry, args.count,
                            time_limit_s=args.time_limit, candidates_per_demand=args.candidates)
        chosen_idx = result["chosen"]
        elapsed = time.perf_counter() - t0
        bound = result["best_bound"]
        gap = "n/a" if bound is None else result["exact_score"] - bound
        print(f"[cp-sat] status={result['status']}  score={result['exact_score']}  "
              f"lower_bound={bound}  gap={gap}  "
              f"proven_optimal={result['proven_optimal']}  time={elapsed:.4f}s")

    chosen_spots = [grid.valid_spots[i] for i in chosen_idx]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.write_solution(chosen_spots, output_path)
    print(f"Saved solution to {output_path}")


if __name__ == "__main__":
    main()
