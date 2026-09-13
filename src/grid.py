"""
Shared grid representation, geometry and scoring for the transformer-placement problem.

All distances in this problem are plain Manhattan distance: every cell on the
grid is traversable (see the original contest rules), so a BFS shortest path
between two points is always identical to their Manhattan distance. There is
no reason to run a graph search at all -- this module never does.
"""
import numpy as np

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Grid:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, "r") as f:
            lines = [line.rstrip("\n") for line in f.readlines() if line.strip("\n") != ""]
        self.rows = len(lines)
        self.cols = max(len(line) for line in lines)
        self.grid = [list(line.ljust(self.cols, "-")) for line in lines]

        self.residential = self._find_all("X")
        self.substations = self._find_all("E")
        self.hospitals = self._find_all("O")
        self.industries = self._find_all("T")
        self.demand_points = self.hospitals + self.industries

        self.valid_spots = self._compute_valid_spots()

    def _find_all(self, symbol):
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] == symbol]

    def _compute_valid_spots(self):
        spots = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != "-":
                    continue
                has_x, has_e = False, False
                for dr, dc in NEIGHBORS_8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if self.grid[nr][nc] == "X":
                            has_x = True
                        elif self.grid[nr][nc] == "E":
                            has_e = True
                if has_x and not has_e:
                    spots.append((r, c))
        return spots

    def apply_solution(self, chosen_spots):
        """Returns a new grid (list of lists of chars) with transformers placed."""
        final = [row[:] for row in self.grid]
        for r, c in chosen_spots:
            final[r][c] = "C"
        return final

    def write_solution(self, chosen_spots, output_path):
        final = self.apply_solution(chosen_spots)
        with open(output_path, "w") as f:
            f.write("\n".join("".join(row) for row in final))


def manhattan_matrix(points_a, points_b):
    """Vectorized pairwise Manhattan distance matrix, shape (len(a), len(b))."""
    if not points_a or not points_b:
        return np.zeros((len(points_a), len(points_b)), dtype=np.int32)
    a = np.array(points_a, dtype=np.int32)
    b = np.array(points_b, dtype=np.int32)
    return np.abs(a[:, None, 0] - b[None, :, 0]) + np.abs(a[:, None, 1] - b[None, :, 1])


def score_configuration(dist_demand_to_spots, chosen_idx):
    """Exact score: sum over demand points of distance to nearest chosen spot.

    dist_demand_to_spots: (num_demand, num_spots) Manhattan distance matrix.
    chosen_idx: iterable of column indices (spots) that are open.
    """
    chosen_idx = list(chosen_idx)
    if not chosen_idx:
        return float("inf")
    sub = dist_demand_to_spots[:, chosen_idx]
    return int(sub.min(axis=1).sum())


def industry_coverage_ok(dist_industry_to_spots, chosen_idx, radius=3, required=2):
    """True if every industry has >= `required` chosen spots within `radius`."""
    if not chosen_idx:
        return dist_industry_to_spots.shape[0] == 0
    sub = dist_industry_to_spots[:, chosen_idx]
    counts = (sub <= radius).sum(axis=1)
    return bool((counts >= required).all())
