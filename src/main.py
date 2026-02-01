import os
import time
import asyncio
import uuid
import random
from collections import deque
from typing import List

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent import FunctionAgent
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool

# BASIC CONFIGURATION
MAP_NAME = "1.txt"
GOOGLE_API_KEY = "AIzaSyCwqIGuy8C_N4nFz6GLm16kuc66Rjw8Bmc"
C_AMOUNT = 28


# ITERATION STRENGTH DEFINES THE NUMBER OF ITERATIONS THE ALGORITHM DOES.
# FOR MAPS OVER 20x20, MORE ITERATIONS TEND TO SECURE ACCURACY.
# THE AI WILL DEFINE THIS VALUE ON ITS OWN BASED ON THE MAP SIZE.

# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# GEMINI CONFIGURATION
llm = GoogleGenAI(model="gemini-2.5-flash", 
                  api_key=GOOGLE_API_KEY,
                  temperature=0.0)
Settings.llm = llm

class MapManager:
    """
    HYBRID SPATIAL OPTIMIZATION ENGINE.

    Architecture:
    1. Pre-calculation: Mapping of valid zones and static distances.
    2. Directed Monte Carlo: Simulation of stochastic scenarios to escape local maxima.
    3. Hill Climbing: Mathematical refinement of the best candidate solution to reach the global optimum.

    This approach eliminates the typical coordinate hallucinations of LLMs.
    """
    def __init__(self, filepath):
        """Initializes the map manager, loading and pre-processing its data."""
        self.filepath = filepath
        with open(filepath, 'r') as f:
            self.grid = [list(line.strip()) for line in f.readlines()]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        # Pre-calculation of key positions to avoid repetitive searches.
        self.targets_O = self._find_all('O')
        self.targets_T = self._find_all('T')
        self.valid_spots = self._get_all_valid_spots()

    def _find_all(self, char_type):
        """Finds all coordinates of a character type in the grid."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] == char_type]

    def _is_valid_spot_logic(self, r, c):
        """Checks if a cell meets the rules for placing a transformer."""
        if self.grid[r][c] != '-': return False
        has_X, has_E = False, False
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] == 'X': has_X = True
                if self.grid[nr][nc] == 'E': has_E = True
        return has_X and not has_E

    def _get_all_valid_spots(self):
        """Pre-calculates and returns a list of all valid coordinates."""
        spots = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self._is_valid_spot_logic(r, c):
                    spots.append((r, c))
        return spots

    def _bfs_dist(self, start, end):
        """Calculates the BFS distance (shortest path) between two points."""
        # Optimization: Manhattan distance first for quick discard
        manhattan = abs(start[0]-end[0]) + abs(start[1]-end[1])
        if manhattan > 10: return manhattan # Quick approximation for far distances
        
        q = deque([(start[0], start[1], 0)])
        visited = {start}
        while q:
            r, c, d = q.popleft()
            if (r, c) == end: return d
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc, d+1))
        return 9999
    
    def _bfs_multi_target(self, start, targets):
        """Calculates the BFS distance from a point to the nearest of several targets."""
        targets_set = set(targets)
        q = deque([(start[0], start[1], 0)])
        visited = {start}
        while q:
            r, c, d = q.popleft()
            if (r, c) in targets_set: return d
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc, d+1))
        return 9999

    def _calculate_score(self, config):
        """Calculates the total 'score' for a configuration, summing distances to 'O' and 'T'."""
        total = 0
        for t in self.targets_O + self.targets_T:
            total += self._bfs_multi_target(t, config)
        return total

    def generate_monte_carlo_solution(self, iteration_strength: int) -> str:
        """
        Generates the best possible solution using a hybrid algorithm approach.
        It does not use AI, it uses statistical brute force and local optimization.

        Process:
        1. Pre-filtering: Identifies all valid spots that are close to industries ('T'),
           using a Manhattan distance <= 3 as a necessary and fast condition.
        2. Directed Monte Carlo Simulation (5000 iterations by default):
           - A. Industry Coverage: For each 'T', ensures that 2 transformers are chosen
             from the pre-filtered list. This guarantees the "Golden Rule".
           - B. Greedy Fill: If there are transformers left to place, it adds them to the
             best remaining spots, prioritizing proximity to hospitals ('O') through
             a Manhattan distance heuristic.
           - C. Validation and Scoring: Each generated configuration is validated with BFS (more precise)
             for the 'T' rule and its total score is calculated. The best one is saved.
        3. Final Polishing (Hill Climbing): The best configuration found in the simulation is
           passed to a local optimization algorithm that tries to move each transformer to its
           neighboring cells to find a local minimum and further reduce the score.
        4. Formatting: The final polished solution is converted into a flat list of coordinates.
        """
        if len(self.valid_spots) < C_AMOUNT:
            return "ERROR: Not enough valid spots on the map."

        best_config = []
        best_score = float('inf')
        
        # 1. Pre-filtering
        spots_near_T = []
        for spot in self.valid_spots:
            for t in self.targets_T:
                if abs(spot[0]-t[0]) + abs(spot[1]-t[1]) <= 3:
                    spots_near_T.append(spot)
                    break
        
        # 2. Simulation Loop
        
        for i in range(iteration_strength):

            # PRINTS A DOT
            if i % 100 == 0:
                print(".", end="", flush=True)

            current_config = []
            valid_T_coverage = True
            
            # STEP A
            required_for_T = set()
            for t in self.targets_T:
                # Search for candidates close to THIS specific T
                candidates = [s for s in spots_near_T if abs(s[0]-t[0]) + abs(s[1]-t[1]) <= 3]
                if len(candidates) < 2:
                    valid_T_coverage = False; break
                
                # Elegimos 2 al azar de los candidatos
                chosen = random.sample(candidates, 2)
                required_for_T.update(chosen)
            
            if not valid_T_coverage: continue
            
            current_config = list(required_for_T)
            
            # If we have chosen too many (due to overlaps), we trim
            if len(current_config) > C_AMOUNT:
                current_config = current_config[:C_AMOUNT]
            
            # STEP B
            while len(current_config) < C_AMOUNT:
                # We choose the best remaining spot for the Hospitals
                remaining_spots = [s for s in self.valid_spots if s not in current_config]
                if not remaining_spots: break
                
                # Quick heuristic: Manhattan distance to the nearest O
                best_filler = min(remaining_spots, key=lambda s: min([abs(s[0]-o[0])+abs(s[1]-o[1]) for o in self.targets_O]))
                current_config.append(best_filler)

            # STEP C
            # Verify real T coverage (BFS)
            bfs_valid = True
            for t in self.targets_T:
                count = sum(1 for c in current_config if self._bfs_dist(t, c) <= 3)
                if count < 2: bfs_valid = False; break
            
            if bfs_valid:
                score = self._calculate_score(current_config)
                if score < best_score:
                    best_score = score
                    best_config = current_config[:]

        # 3. FINAL POLISHING (Hill Climbing)
        # We already have the best seed, now we optimize it to the pixel
        print("\n\nREFINING BEST SOLUTION (HILL CLIMBING)...", flush=True)
        final_polished = self._hill_climbing(best_config, iteration_strength)
        
        # Format flat output
        flat = []
        for r, c in final_polished: flat.extend([r, c])
        return str(flat)

    def _hill_climbing(self, config, iteration_strength: int):
        """
        Final local optimization using the Hill Climbing algorithm.

        This method takes a transformer configuration and tries to improve it
        iteratively by making small local changes. The goal is to find
        a local minimum in the solution "landscape", where the "height" is the
        score.

        Process:
        1. It starts from an initial configuration (`config`).
        2. It iterates a fixed number of times (or until there is no improvement).
        3. In each iteration, it tries to move one transformer at a time to one of
           its 8 neighboring cells.
        4. For each potential move, cascaded validations are performed:
           a. The new site must be a valid spot and not already occupied.
           b. (Quick Check) The 'T' rule is checked with Manhattan distance.
              If it fails, the move is discarded immediately.
           c. (Expensive Check) If the quick check passes, the full score is calculated
              with BFS. If the score does not improve, it is discarded.
           d. (Final Check) If the score improves, the rule of
              the 'T's is re-verified with the expensive but precise BFS.
        5. If a move passes all validations and improves the score, it is
           accepted, and the improvement search process restarts from this
           new and better configuration.
        6. The algorithm ends when no improvement is found in a
           full iteration, which means a local minimum has been reached.
        """
        current = config[:]
        best_local_score = self._calculate_score(current)
        
        # Use a fraction of the main iteration strength for polishing, with a minimum.
        hill_climbing_attempts = max(500, int(iteration_strength / 10))
        for _ in range(hill_climbing_attempts): # improvement attempts
            improved = False
            for i in range(len(current)):
                orig = current[i]
                r, c = orig
                neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1), (r-1,c-1), (r-1,c+1), (r+1,c-1), (r+1,c+1)]
                for nr, nc in neighbors:
                    if (nr, nc) in self.valid_spots and (nr, nc) not in current:
                        # Test change
                        new_cfg = current[:]
                        new_cfg[i] = (nr, nc)
                        
                        # Validate T quickly
                        valid_T = True
                        for t in self.targets_T:
                            if sum(1 for x in new_cfg if abs(x[0]-t[0])+abs(x[1]-t[1]) <= 3) < 2: # Quick Manhattan check
                                valid_T = False; break # If Manhattan fails, BFS will surely fail
                        
                        if valid_T:
                            # Expensive real Score check
                            new_score = self._calculate_score(new_cfg)
                            if new_score < best_local_score:
                                # Final strict T check (BFS)
                                valid_T_strict = True
                                for t in self.targets_T:
                                    if sum(1 for x in new_cfg if self._bfs_dist(t, x) <= 3) < 2:
                                        valid_T_strict = False; break
                                
                                # Saves the best score
                                if valid_T_strict:
                                    current = new_cfg
                                    best_local_score = new_score
                                    improved = True
                                    break
                if improved: break
            if not improved: break
        return current

    def save_solution(self, coords_list: List[int]):
        """Saves the final configuration to an output file, respecting the format."""
        output_path = os.path.join(OUTPUT_DIR, MAP_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_grid = [row[:] for row in self.grid]
        for i in range(0, len(coords_list), 2):
            r, c = coords_list[i], coords_list[i+1]
            final_grid[r][c] = 'C'
        with open(output_path, 'w') as f:
            f.write("\n".join("".join(row) for row in final_grid))
        return f"Saved to {output_path} with approximate Score."

# TOOLS
map_manager = MapManager(os.path.join(INPUT_DIR, MAP_NAME))

def generate_master_solution(iteration_strength: int) -> str:
    """
    Generates simulations, chooses the best one, and polishes it.
    Returns the final list of coordinates ready to be saved.

    :param iteration_strength: The number of Monte Carlo simulations to run. A good baseline is 5000. For larger maps, a higher value is recommended for better results.
    """
    print(f"\nAGENT HAS DECIDED TO PERFORM {iteration_strength} ITERATIONS.\n")
    return map_manager.generate_monte_carlo_solution(iteration_strength)

def save_file(coords_list: List[int]) -> str:
    return map_manager.save_solution(coords_list)

tools = [
    FunctionTool.from_defaults(fn=generate_master_solution),
    FunctionTool.from_defaults(fn=save_file)
]

# MAIN
async def main():
    agent = FunctionAgent(tools=tools, llm=llm, verbose=True)
    #Gemini (LLM): Is the Manager. It understands the problem and gives the order, but does not do the dirty work.
    #Python (Algorithms): Is the Engineer. It executes massive mathematical calculations that the AI cannot do.
    run_id = str(uuid.uuid4())
    prompt = f"""
    ID: {run_id}
    OBJECTIVE: Solve the map by placing {C_AMOUNT} transformers. The map size is {map_manager.rows}x{map_manager.cols}.
    
    1. First, decide on an appropriate `iteration_strength` for the `generate_master_solution` tool. A good baseline is 5000. For maps larger than 15x15, you should increase this value proportionately to get a better result (e.g., 10000 or more for very large maps).
    2. Execute `generate_master_solution` with your chosen `iteration_strength`. This tool will do ALL the mathematical work.
    3. With the coordinate data the first tool returns, execute `save_file`.
    
    Think step-by-step. First, call `generate_master_solution` with a smart `iteration_strength`, then call `save_file` with the result.
    """

    try:
        print("GENERATING ALL POSSIBLE COMBINATIONS...")
        await agent.run(user_msg=prompt)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    t0 = time.perf_counter()

    print("\nCURRENT INPUT MAP:\n")
    for row in map_manager.grid:
        print("".join(row))

    print("\n")

    print(f"ATTENTION! YOU ARE LOOKING FOR - {C_AMOUNT} - 'C' TRANSFORMERS\n")

    asyncio.run(main())
    print(f"\nSOLUTION FOUND!")
    print(f"\nTOTAL TIME: {time.perf_counter() - t0:.4f}s")