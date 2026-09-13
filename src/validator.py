"""
Strict rule-compliance checker and scorer for a generated solution.

Distance is plain Manhattan distance: the grid has no obstacles (every cell
is traversable), so a BFS shortest path is always identical to Manhattan
distance -- there's no need to run a graph search here.
"""
import argparse
import os

from grid import Grid, manhattan_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "inputs")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FIXED_ELEMENTS = {"O", "X", "E", "T"}
NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def verify_positions(orig, gen):
    if len(orig) != len(gen) or len(orig[0]) != len(gen[0]):
        print("Error: The maps have different dimensions")
        return False
    for i in range(len(orig)):
        for j in range(len(orig[0])):
            if orig[i][j] in FIXED_ELEMENTS or gen[i][j] in FIXED_ELEMENTS:
                if orig[i][j] != gen[i][j]:
                    print(f"Error at position ({i},{j}): Original '{orig[i][j]}' vs Generated '{gen[i][j]}'")
                    return False
    return True


def verify_transformer_count(gen, expected_count):
    count = sum(row.count("C") for row in gen)
    if count != expected_count:
        print(f"Error: Expected {expected_count} 'C' transformers, but found {count}")
        return False
    return True


def verify_c_in_blank_spaces(orig, gen):
    rows, cols = len(orig), len(orig[0])
    for i in range(rows):
        for j in range(cols):
            if gen[i][j] == "C" and orig[i][j] != "-":
                print(f"Error: Transformer 'C' at ({i},{j}) is not in a blank space (original: '{orig[i][j]}')")
                return False
    return True


def verify_c_has_adjacent_x(gen):
    rows, cols = len(gen), len(gen[0])
    for i in range(rows):
        for j in range(cols):
            if gen[i][j] != "C":
                continue
            if not any(0 <= i + di < rows and 0 <= j + dj < cols and gen[i + di][j + dj] == "X"
                       for di, dj in NEIGHBORS_8):
                print(f"Error: Transformer 'C' at ({i},{j}) has no adjacent 'X' (residential area)")
                return False
    return True


def verify_c_without_adjacent_e(gen):
    rows, cols = len(gen), len(gen[0])
    for i in range(rows):
        for j in range(cols):
            if gen[i][j] != "C":
                continue
            for di, dj in NEIGHBORS_8:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and gen[ni][nj] == "E":
                    print(f"Error: Transformer 'C' at ({i},{j}) has adjacent 'E' (Substation) at ({ni},{nj})")
                    return False
    return True


def verify_t_has_nearby_c(gen, radius=3, required=2):
    rows, cols = len(gen), len(gen[0])
    towers = [(i, j) for i in range(rows) for j in range(cols) if gen[i][j] == "T"]
    transformers = [(i, j) for i in range(rows) for j in range(cols) if gen[i][j] == "C"]
    if not towers:
        return True
    dist = manhattan_matrix(towers, transformers)
    counts = (dist <= radius).sum(axis=1)
    for tower, count in zip(towers, counts):
        if count < required:
            print(f"Error: Tower 'T' at {tower} has only {count} 'C' transformers within radius {radius} "
                  f"(minimum required: {required})")
            return False
    return True


def total_distance(gen):
    rows, cols = len(gen), len(gen[0])
    houses = [(i, j) for i in range(rows) for j in range(cols) if gen[i][j] == "O"]
    towers = [(i, j) for i in range(rows) for j in range(cols) if gen[i][j] == "T"]
    transformers = [(i, j) for i in range(rows) for j in range(cols) if gen[i][j] == "C"]
    dist = manhattan_matrix(houses + towers, transformers)
    return int(dist.min(axis=1).sum())


def validate(original_map, generated_map, expected_count):
    checks = [
        verify_positions(original_map, generated_map),
        verify_transformer_count(generated_map, expected_count),
        verify_c_in_blank_spaces(original_map, generated_map),
        verify_c_has_adjacent_x(generated_map),
        verify_c_without_adjacent_e(generated_map),
        verify_t_has_nearby_c(generated_map),
    ]
    if not all(checks):
        return None
    return total_distance(generated_map)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="1.txt", help="Map filename inside inputs/, or a full path.")
    p.add_argument("--output", default=None, help="Generated map path (default: outputs/<input filename>).")
    p.add_argument("--count", type=int, default=28, help="Expected number of 'C' transformers.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input if os.path.isabs(args.input) or os.path.exists(args.input) else os.path.join(DEFAULT_INPUT_DIR, args.input)
    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, os.path.basename(input_path))

    with open(input_path) as f:
        original_map = [list(line.strip()) for line in f.readlines()]
    with open(output_path) as f:
        generated_map = [list(line.strip()) for line in f.readlines()]

    score = validate(original_map, generated_map, args.count)
    if score is None:
        print("Error: The generated map does not meet the requirements")
    else:
        print(f"Final Score: {score}")
