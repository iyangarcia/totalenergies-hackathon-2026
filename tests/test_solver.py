import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grid import Grid, manhattan_matrix, score_configuration, industry_coverage_ok
from solver import greedy_construct, solve_ilp

MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "inputs", "1.txt")
COUNT = 28


def load():
    grid = Grid(MAP_PATH)
    dist_demand = manhattan_matrix(grid.demand_points, grid.valid_spots)
    dist_industry = manhattan_matrix(grid.industries, grid.valid_spots)
    return grid, dist_demand, dist_industry


def assert_valid_placement(grid, chosen_idx, k):
    assert len(chosen_idx) == k
    assert len(set(chosen_idx)) == k  # no duplicates

    chosen_spots = [grid.valid_spots[i] for i in chosen_idx]
    for r, c in chosen_spots:
        assert grid.grid[r][c] == "-"  # only placed on originally-empty cells

    valid_set = set(grid.valid_spots)
    for spot in chosen_spots:
        assert spot in valid_set  # respects residential/substation adjacency rules


def test_manhattan_matrix_matches_bfs_on_open_grid():
    # Ground truth: BFS on a grid with zero obstacles must equal Manhattan distance.
    from collections import deque

    grid = Grid(MAP_PATH)

    def bfs(start, end):
        q = deque([(start[0], start[1], 0)])
        visited = {start}
        while q:
            r, c, d = q.popleft()
            if (r, c) == end:
                return d
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc, d + 1))
        return None

    import random
    random.seed(0)
    sample_points = grid.valid_spots[:15]
    for a in sample_points:
        for b in sample_points[:5]:
            assert bfs(a, b) == abs(a[0] - b[0]) + abs(a[1] - b[1])


def test_greedy_produces_valid_and_covering_solution():
    grid, dist_demand, dist_industry = load()
    chosen_idx = greedy_construct(grid, dist_demand, dist_industry, COUNT)
    assert_valid_placement(grid, chosen_idx, COUNT)
    assert industry_coverage_ok(dist_industry, chosen_idx)


def test_ilp_produces_valid_and_covering_solution():
    grid, dist_demand, dist_industry = load()
    result = solve_ilp(grid, dist_demand, dist_industry, COUNT, time_limit_s=10.0)
    assert_valid_placement(grid, result["chosen"], COUNT)
    assert industry_coverage_ok(dist_industry, result["chosen"])
    assert result["status"] in ("OPTIMAL", "FEASIBLE")


def test_ilp_is_at_least_as_good_as_greedy():
    grid, dist_demand, dist_industry = load()
    greedy_idx = greedy_construct(grid, dist_demand, dist_industry, COUNT)
    greedy_score = score_configuration(dist_demand, greedy_idx)

    result = solve_ilp(grid, dist_demand, dist_industry, COUNT, time_limit_s=10.0)
    assert result["exact_score"] <= greedy_score


def test_ilp_proves_optimality_on_small_map():
    grid, dist_demand, dist_industry = load()
    result = solve_ilp(grid, dist_demand, dist_industry, COUNT, time_limit_s=10.0)
    assert result["proven_optimal"]
    assert result["exact_score"] == result["best_bound"]


def test_infeasible_when_not_enough_valid_spots():
    grid, dist_demand, dist_industry = load()
    too_many = len(grid.valid_spots) + 1
    try:
        solve_ilp(grid, dist_demand, dist_industry, too_many, time_limit_s=5.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
