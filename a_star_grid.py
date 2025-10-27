import random, math, time, os
from heapq import heappush, heappop
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Grid & Utilities -----------------------
class Grid:
    def __init__(self, rows=30, cols=30, obstacle_prob=0.28, seed=None):
        self.rows, self.cols = rows, cols
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.grid = (np.random.rand(rows, cols) < obstacle_prob).astype(np.int8)
        # Place start/goal on free cells
        self.start = self._rand_free()
        self.goal = self._rand_free()
        while self.goal == self.start:
            self.goal = self._rand_free()
    def _rand_free(self):
        while True:
            r, c = random.randrange(self.rows), random.randrange(self.cols)
            if self.grid[r, c] == 0:
                return (r, c)
    def neighbors4(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                yield (nr, nc)

# ----------------------- Heuristics -----------------------------
def h_manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def h_euclidean(a, b):
    dx, dy = a[0]-b[0], a[1]-b[1]
    return math.hypot(dx, dy)

def h_chebyshev(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

HEURISTICS = {
    'manhattan': h_manhattan,
    'euclidean': h_euclidean,
    'chebyshev': h_chebyshev,
}

# ----------------------- A* Search ------------------------------
class AStarResult:
    __slots__ = ("success","path","path_length","nodes_expanded","runtime_ms","ebf")
    def __init__(self, **kw):
        for k,v in kw.items(): setattr(self, k, v)

def astar(grid: Grid, start, goal, heuristic):
    t0 = time.perf_counter()
    open_heap = []
    heappush(open_heap, (0, 0, start))  # (f, tie, node)
    g = {start: 0}
    parent = {start: None}
    nodes_expanded = 0
    tie = 0
    while open_heap:
        f, _, u = heappop(open_heap)
        nodes_expanded += 1
        if u == goal:
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            runtime_ms = (time.perf_counter() - t0)*1000
            # Effective branching factor (rough): N ≈ 1 + b + b^2 + ... + b^d
            d = len(path)-1
            N = nodes_expanded
            ebf = estimate_ebf(N, max(d,1))
            return AStarResult(success=True, path=path, path_length=d, nodes_expanded=N, runtime_ms=runtime_ms, ebf=ebf)
        for v in grid.neighbors4(*u):
            alt = g[u] + 1
            if alt < g.get(v, float('inf')):
                g[v] = alt
                tie += 1
                heappush(open_heap, (alt + heuristic(v, goal), tie, v))
                parent[v] = u
    runtime_ms = (time.perf_counter() - t0)*1000
    return AStarResult(success=False, path=[], path_length=None, nodes_expanded=nodes_expanded, runtime_ms=runtime_ms, ebf=None)

def estimate_ebf(N, d, iters=20):
    # Solve N = (b^{d+1}-1)/(b-1) for b via binary search (b>1)
    if d <= 0 or N <= 1: return 0.0
    lo, hi = 1.0001, 50.0
    for _ in range(iters):
        mid = (lo+hi)/2
        total = (mid**(d+1)-1)/(mid-1)
        if total < N:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2

# ----------------------- Experiment Harness --------------------
def run_trials(rows=30, cols=30, obstacle_prob=0.28, trials=50, seed=None):
    rng = random.Random(seed)
    results = {name: [] for name in HEURISTICS}
    for t in range(trials):
        g = Grid(rows, cols, obstacle_prob, seed=rng.randrange(10**9))
        for name, hfun in HEURISTICS.items():
            r = astar(g, g.start, g.goal, hfun)
            results[name].append(r)
    return results

def summarize(results):
    summary = {}
    for name, runs in results.items():
        succ = [r for r in runs if r.success]
        success_rate = len(succ)/len(runs)
        avg_len = np.mean([r.path_length for r in succ]) if succ else float('nan')
        avg_nodes = np.mean([r.nodes_expanded for r in runs])
        avg_rt = np.mean([r.runtime_ms for r in runs])
        avg_ebf = np.mean([r.ebf for r in succ]) if succ else float('nan')
        summary[name] = dict(success_rate=success_rate, avg_path_length=avg_len,
                             avg_nodes_expanded=avg_nodes, avg_runtime_ms=avg_rt, avg_ebf=avg_ebf)
    return summary

def plot_bar(summary, metric, out_png):
    os.makedirs('results', exist_ok=True)
    labels = list(summary.keys())
    values = [summary[k][metric] for k in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.ylabel(metric)
    plt.title(f"Heuristic comparison – {metric}")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def show_grid(grid, path=None):
    import matplotlib.pyplot as plt
    data = grid.grid.copy()
    sr, sc = grid.start
    gr, gc = grid.goal
    data[sr, sc] = 2
    data[gr, gc] = 3
    plt.figure(figsize=(6,6))
    plt.imshow(data, cmap='gray_r')
    if path:
        pr, pc = zip(*path)
        plt.plot(pc, pr, 'r-', linewidth=2)
    plt.title("A* Grid Path (red)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    results = run_trials(rows=40, cols=40, obstacle_prob=0.30, trials=80, seed=42)
    summary = summarize(results)
    print("\nA* Heuristic Comparison (mean over runs):")
    for k, v in summary.items():
        print(f"- {k:10s} | success={v['success_rate']:.2f} | path_len={v['avg_path_length']:.2f} | "
              f"nodes={v['avg_nodes_expanded']:.1f} | rt_ms={v['avg_runtime_ms']:.2f} | ebf={v['avg_ebf']:.3f}")
    # Plots
    plot_bar(summary, 'success_rate', 'results/astar_success_rate.png')
    plot_bar(summary, 'avg_nodes_expanded', 'results/astar_nodes_expanded.png')
    plot_bar(summary, 'avg_runtime_ms', 'results/astar_runtime.png')
    plot_bar(summary, 'avg_path_length', 'results/astar_path_length.png')
    example_grid = Grid(rows=40, cols=40, obstacle_prob=0.3, seed=5)
    res = astar(example_grid, example_grid.start, example_grid.goal, h_manhattan)
    if res.success:
        show_grid(example_grid, res.path)
