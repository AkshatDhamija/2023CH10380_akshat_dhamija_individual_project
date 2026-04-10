"""
simulate.py
-----------
BTW sandpile simulation on the rock music influence network.

Each node = a rock artist / band
Each grain = one unit of accumulated cultural influence
Threshold  = artist's out-degree (number they directly influenced)
Toppling   = cultural cascade: influence spreads to all successors
Avalanche  = full cascade size (total topplings until stability)

Usage (standalone):
    python src/simulate.py               # full run
    python src/simulate.py --quick       # 32-node test, 50k steps
    python src/simulate.py --sweep       # connectivity sweep only
"""

import os
import sys
import argparse
import time
import numpy as np

# Allow running as a script from any directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.network import build_graph, get_thresholds

# ── Output paths ───────────────────────────────────────────────
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Default simulation parameters ─────────────────────────────
DEFAULT_T    = 500_000   # number of cultural shock events
DEFAULT_SEED = 42        # for reproducibility
SWEEP_LAMS   = [0.5, 0.75, 1.0, 1.5, 2.0]   # threshold scaling values


# ══════════════════════════════════════════════════════════════
#  Core avalanche engine
# ══════════════════════════════════════════════════════════════
def run_avalanche(heights: dict, thresholds: dict,
                 successors: dict) -> int:
    """
    Propagate one avalanche to completion using parallel toppling.

    Parameters
    ----------
    heights    : mutable dict {node -> current influence score}
    thresholds : dict {node -> toppling threshold (= out-degree)}
    successors : dict {node -> list of successor nodes}

    Returns
    -------
    size : int
        Total number of toppling events (= avalanche size s).
    """
    size = 0

    while True:
        # Identify all currently unstable nodes
        # (skip leaf nodes whose threshold > their out-degree = 0)
        unstable = [
            v for v, h in heights.items()
            if h >= thresholds[v] and len(successors[v]) > 0
        ]

        if not unstable:
            break

        size += len(unstable)

        # Topple all unstable nodes simultaneously (parallel update)
        for v in unstable:
            heights[v] -= thresholds[v]        # discharge
            for u in successors[v]:
                heights[u] += 1                # distribute to successors

    return size


# ══════════════════════════════════════════════════════════════
#  Main simulation
# ══════════════════════════════════════════════════════════════
def simulate(T: int = DEFAULT_T,
             lam: float = 1.0,
             seed: int = DEFAULT_SEED,
             save: bool = True,
             verbose: bool = True) -> np.ndarray:
    """
    Run the full BTW sandpile simulation on the rock influence network.

    Parameters
    ----------
    T       : number of cultural shock (grain drop) events
    lam     : threshold scaling factor (1.0 = standard, <1 = denser)
    seed    : random seed for reproducibility
    save    : whether to save results to disk
    verbose : whether to print progress

    Returns
    -------
    avalanche_sizes : np.ndarray of int64
        All non-zero avalanche sizes recorded during simulation.
    """
    # ── Build network ─────────────────────────────────────────
    G          = build_graph()
    nodes      = list(G.nodes())
    thresholds = get_thresholds(G, lam=lam)

    # Pre-compute successor lists for speed
    successors = {v: list(G.successors(v)) for v in nodes}

    # ── Initialise ────────────────────────────────────────────
    rng     = np.random.default_rng(seed)
    heights = {v: 0 for v in nodes}

    avalanche_sizes = []
    t_start = time.time()

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Rock SOC Simulation")
        print(f"  Nodes: {len(nodes)}  |  Edges: {G.number_of_edges()}")
        print(f"  T={T:,}  |  lambda={lam}  |  seed={seed}")
        print(f"{'='*55}")

    # ── Main loop ─────────────────────────────────────────────
    for t in range(T):
        # NOISE: drop one grain at a random node
        v = rng.choice(nodes)
        heights[v] += 1

        # AVALANCHE: propagate until stable
        s = run_avalanche(heights, thresholds, successors)
        if s > 0:
            avalanche_sizes.append(s)

        # Progress report every 100k steps
        if verbose and (t + 1) % 100_000 == 0:
            pct = (t + 1) / T * 100
            n_av = len(avalanche_sizes)
            mx   = max(avalanche_sizes) if avalanche_sizes else 0
            elapsed = time.time() - t_start
            print(f"  {pct:5.1f}%  |  avalanches: {n_av:,}"
                  f"  |  max size: {mx:,}"
                  f"  |  elapsed: {elapsed:.1f}s")

    sizes = np.array(avalanche_sizes, dtype=np.int64)

    if verbose:
        print(f"\n  Done in {time.time()-t_start:.1f}s")
        print(f"  Total avalanches : {len(sizes):,}")
        print(f"  Max size         : {sizes.max():,}")
        print(f"  Mean size        : {sizes.mean():.3f}")
        print(f"  Median size      : {np.median(sizes):.0f}")
        print(f"{'='*55}\n")

    # ── Save results ──────────────────────────────────────────
    if save:
        tag  = f"lam{str(lam).replace('.','p')}"
        path = os.path.join(RESULTS_DIR, f'avalanche_sizes_{tag}.npy')
        np.save(path, sizes)

        # Also save final heights (grid state)
        heights_arr = np.array([heights[v] for v in nodes])
        hpath = os.path.join(RESULTS_DIR, f'final_heights_{tag}.npy')
        np.save(hpath, heights_arr)

        # Save node order for reference
        npath = os.path.join(RESULTS_DIR, 'node_order.txt')
        with open(npath, 'w') as f:
            f.write('\n'.join(nodes))

        if verbose:
            print(f"  Saved: {path}")
            print(f"  Saved: {hpath}")

    return sizes


# ══════════════════════════════════════════════════════════════
#  Connectivity sweep
# ══════════════════════════════════════════════════════════════
def connectivity_sweep(T: int = DEFAULT_T,
                       lams: list = None,
                       seed: int = DEFAULT_SEED) -> dict:
    """
    Repeat simulation for each lambda in lams.
    Returns dict {lambda -> avalanche_sizes array}.
    Results saved to results/connectivity_sweep.npz
    """
    if lams is None:
        lams = SWEEP_LAMS

    results = {}
    print(f"\n{'='*55}")
    print(f"  Connectivity Sweep  |  lambdas: {lams}")
    print(f"{'='*55}")

    for lam in lams:
        print(f"\n  >> lambda = {lam}")
        sizes = simulate(T=T, lam=lam, seed=seed,
                         save=True, verbose=True)
        results[lam] = sizes

    # Save all sweep results in one file
    sweep_path = os.path.join(RESULTS_DIR, 'connectivity_sweep.npz')
    np.savez(
        sweep_path,
        **{f'lam_{str(lam).replace(".", "p")}': v
           for lam, v in results.items()}
    )
    print(f"\n  Sweep saved -> {sweep_path}")
    return results


# ══════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rock Music SOC Simulation')
    parser.add_argument('--quick', action='store_true',
        help='Fast test run: T=50,000 steps')
    parser.add_argument('--sweep', action='store_true',
        help='Run connectivity sweep across all lambda values')
    parser.add_argument('--T', type=int, default=DEFAULT_T,
        help=f'Number of steps (default: {DEFAULT_T:,})')
    parser.add_argument('--lam', type=float, default=1.0,
        help='Threshold scaling factor (default: 1.0)')
    args = parser.parse_args()

    if args.sweep:
        connectivity_sweep(T=args.T if not args.quick else 50_000)
    else:
        T = 50_000 if args.quick else args.T
        simulate(T=T, lam=args.lam)
