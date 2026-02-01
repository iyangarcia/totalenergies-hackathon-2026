from collections import deque
import os

# --- GLOBAL VARIABLES ---
MAP_NAME = "1.txt"
# Expected number of 'C' transformers
C_AMOUNT = 28

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# --- 1. Read original and generated maps ---
input_path = os.path.join(BASE_DIR, "inputs", MAP_NAME)
output_path = os.path.join(BASE_DIR, "outputs", MAP_NAME)

with open(input_path, "r") as f:
    original_map = [list(line.strip()) for line in f.readlines()]

with open(output_path, "r") as f:
    generated_map = [list(line.strip()) for line in f.readlines()]

# --- 2. Verify that O, X, E, and T are in the same positions ---
# Checks that no 'E', 'T', or 'O' have been removed from the original map
def verify_positions(orig_map, gen_map):
    orig_rows = len(orig_map)
    orig_cols = len(orig_map[0]) if orig_rows > 0 else 0
    gen_rows = len(gen_map)
    gen_cols = len(gen_map[0]) if gen_rows > 0 else 0
    
    # Verify dimensions
    if orig_rows != gen_rows or orig_cols != gen_cols:
        print(f"Error: The maps have different dimensions")
        return False
    
    # Verify positions of O, X, E, and T (elements that should not change)
    fixed_elements = ['O', 'X', 'E', 'T']
    for i in range(orig_rows):
        for j in range(orig_cols):
            # If the original has O, X, E, or T, it must be the same in the generated one
            if orig_map[i][j] in fixed_elements:
                if gen_map[i][j] != orig_map[i][j]:
                    print(f"Error at position ({i},{j}): Original '{orig_map[i][j]}' vs Generated '{gen_map[i][j]}'")
                    return False
            # If the generated one has O, X, E, or T, it must be the same as in the original
            if gen_map[i][j] in fixed_elements:
                if orig_map[i][j] != gen_map[i][j]:
                    print(f"Error at position ({i},{j}): Original '{orig_map[i][j]}' vs Generated '{gen_map[i][j]}'")
                    return False
    
    return True

# --- 3. Verify number of 'C' transformers ---
def verify_transformer_count(map_grid, expected_count):
    transformers = 0
    for row in map_grid:
        for cell in row:
            if cell == 'C':
                transformers += 1
    
    if transformers != expected_count:
        print(f"Error: Expected {expected_count} 'C' transformers, but found {transformers}")
        return False
    
    return True

# --- 3.1 Verify that all 'C's were placed where there was a '-' before ---
def verify_c_in_blank_spaces(orig_map, gen_map):
    rows = len(orig_map)
    cols = len(orig_map[0]) if rows > 0 else 0
    
    for i in range(rows):
        for j in range(cols):
            if gen_map[i][j] == 'C':
                if orig_map[i][j] != '-':
                    print(f"Error: Transformer 'C' at position ({i},{j}) is not in a blank space '-' (original: '{orig_map[i][j]}')")
                    return False
    
    return True

# --- 3.2 Verify that each 'C' has at least one 'X' in the 8 adjacent cells ---
def verify_c_has_adjacent_x(map_grid):
    rows = len(map_grid)
    cols = len(map_grid[0]) if rows > 0 else 0
    
    # Directions of the 8 adjacent cells (includes diagonals)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for i in range(rows):
        for j in range(cols):
            if map_grid[i][j] == 'C':
                has_adjacent_x = False
                
                # Check the 8 adjacent cells
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    # Check that it is within the limits
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if map_grid[ni][nj] == 'X':
                            has_adjacent_x = True
                            break
                
                if not has_adjacent_x:
                    print(f"Error: Transformer 'C' at position ({i},{j}) does not have any 'X' (residential area) in the 8 adjacent cells")
                    return False
    
    return True

# --- 3.3 Verify that each 'C' does NOT have any 'E' in the 8 adjacent cells ---
def verify_c_without_adjacent_e(map_grid):
    rows = len(map_grid)
    cols = len(map_grid[0]) if rows > 0 else 0
    
    # Directions of the 8 adjacent cells (includes diagonals)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for i in range(rows):
        for j in range(cols):
            if map_grid[i][j] == 'C':
                # Check the 8 adjacent cells
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    # Check that it is within the limits
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if map_grid[ni][nj] == 'E':
                            print(f"Error: Transformer 'C' at position ({i},{j}) has an adjacent 'E' (Substation) at position ({ni},{nj})")
                            return False
    
    return True

# --- 3.4 Verify that all 'T's have at least 2 'C's within a distance of 3 cells ---
def verify_t_has_nearby_c(map_grid):
    rows = len(map_grid)
    cols = len(map_grid[0]) if rows > 0 else 0
    
    towers = [(i, j) for i in range(rows) for j in range(cols) if map_grid[i][j] == 'T']
    transformers = [(i, j) for i in range(rows) for j in range(cols) if map_grid[i][j] == 'C']
    
    for tower in towers:
        # Count transformers within a radius of 3
        nearby_transformers = 0
        for transformer in transformers:
            distance = minimum_distance(map_grid, tower, [transformer])
            if distance <= 3:
                nearby_transformers += 1
        
        if nearby_transformers < 2:
            print(f"Error: Tower 'T' at position {tower} has only {nearby_transformers} 'C' transformers within a radius of 3 cells (minimum required: 2)")
            return False
    
    return True

# --- 4. Function to calculate minimum distance ---
def minimum_distance(map_grid, start, targets):
    rows, cols = len(map_grid), len(map_grid[0])
    visited = [[False]*cols for _ in range(rows)]
    queue = deque([(start[0], start[1], 0)])
    
    while queue:
        i, j, d = queue.popleft()
        if (i, j) in targets:
            return d
        if visited[i][j]:
            continue
        visited[i][j] = True
        
        # Move in the 4 directions (up, down, left, right)
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            # Just check that it is within the limits
            # ALL cells are traversable (-, X, O, C, T, E)
            if 0 <= ni < rows and 0 <= nj < cols:
                queue.append((ni, nj, d+1))
    
    return float('inf')

# --- 5. Total distance function ---
# Calculates the sum of minimum distances:
# - From each 'O' (house/hospital) to the nearest 'C' (transformer)
# - From each 'T' (tower/industry) to the nearest 'C' (transformer)
# NOTE: 'X' (residential areas) are only traversable, NOT counted in the calculation
def total_distance(map_grid):
    rows, cols = len(map_grid), len(map_grid[0])
    houses = [(i,j) for i in range(rows) for j in range(cols) if map_grid[i][j] == 'O']
    towers = [(i,j) for i in range(rows) for j in range(cols) if map_grid[i][j] == 'T']
    transformers = [(i,j) for i in range(rows) for j in range(cols) if map_grid[i][j] == 'C']
    
    total = 0
    # Distance from each house O to the nearest C
    for house in houses:
        total += minimum_distance(map_grid, house, transformers)
    # Distance from each tower T to the nearest C
    for tower in towers:
        total += minimum_distance(map_grid, tower, transformers)
    return total

# --- 6. Run verifications and calculation ---
if (verify_positions(original_map, generated_map) and 
    verify_transformer_count(generated_map, C_AMOUNT) and
    verify_c_in_blank_spaces(original_map, generated_map) and
    verify_c_has_adjacent_x(generated_map) and
    verify_c_without_adjacent_e(generated_map) and
    verify_t_has_nearby_c(generated_map)):
    total_steps = total_distance(generated_map)
    print(f"Final Score: {total_steps}")
else:
    print("Error: The generated map does not meet the requirements")