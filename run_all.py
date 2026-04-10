"""
run_all.py
----------
Single entry point for the Rock Music SOC project.
Runs the full pipeline: simulate -> analyze -> report summary.

Usage
-----
    python run_all.py                  # full run (recommended)
    python run_all.py --quick          # fast test (~1 min)
    python run_all.py --sweep          # include connectivity sweep
    python run_all.py --analyze-only   # skip simulation, plot only

"""

import argparse
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def print_banner():
    print()
    print('╔══════════════════════════════════════════════════════╗')
    print('║   Rock Music Influence Network — SOC Simulation      ║')
    print('║   Complexity Science Individual Project               ║')
    print('║   IIT Delhi, 2026                                     ║')
    print('╚══════════════════════════════════════════════════════╝')
    print()


def check_dependencies():
    """Check all required packages are installed."""
    required = {
        'numpy'      : 'numpy',
        'matplotlib' : 'matplotlib',
        'scipy'      : 'scipy',
        'networkx'   : 'networkx',
    }
    missing = []
    for pkg, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print('  [ERROR] Missing packages:', ', '.join(missing))
        print('  Install with:')
        print(f'    pip install {" ".join(missing)}')
        sys.exit(1)
    else:
        print('  [✓] All dependencies satisfied')


def step_network():
    """Step 0: Build and validate the network."""
    print('\n' + '─' * 55)
    print('  STEP 0 — Building and validating influence network')
    print('─' * 55)
    from src.network import build_graph, print_network_stats, \
        plot_network, plot_degree_distribution
    G = build_graph()
    print_network_stats(G)
    plot_network(G)
    plot_degree_distribution(G)
    return G


def step_simulate(quick: bool = False, sweep: bool = False,
                  T: int = 500_000):
    """Step 1: Run BTW sandpile simulation."""
    print('\n' + '─' * 55)
    print('  STEP 1 — Running BTW sandpile simulation')
    print('─' * 55)
    from src.simulate import simulate, connectivity_sweep

    t0 = time.time()

    if quick:
        print('  [MODE] Quick run  (T = 50,000 steps)')
        simulate(T=50_000, lam=1.0, verbose=True)
    else:
        print(f'  [MODE] Full run  (T = {T:,} steps)')
        simulate(T=T, lam=1.0, verbose=True)

    if sweep:
        print('\n  [MODE] Running connectivity sweep...')
        connectivity_sweep(T=50_000 if quick else T)

    elapsed = time.time() - t0
    print(f'\n  [✓] Simulation complete in {elapsed/60:.1f} min')


def step_analyze():
    """Step 2: Generate all figures and power-law fit."""
    print('\n' + '─' * 55)
    print('  STEP 2 — Running power-law analysis and generating figures')
    print('─' * 55)
    from src.analyze import run_analysis
    run_analysis()
    print('  [✓] Analysis complete')


def print_summary():
    """Print final submission checklist."""
    results_dir = os.path.join(ROOT, 'results', 'figures')
    figs = []
    if os.path.exists(results_dir):
        figs = [f for f in os.listdir(results_dir)
                if f.endswith('.png')]

    print()
    print('╔══════════════════════════════════════════════════════╗')
    print('║   DONE — Submission Checklist                        ║')
    print('╠══════════════════════════════════════════════════════╣')
    print(f'║   Figures generated : {len(figs):<5}                         ║')
    print('║                                                      ║')
    print('║   To compile the report:                             ║')
    print('║     cd report && pdflatex main.tex                   ║')
    print('║              && pdflatex main.tex                    ║')
    print('║                                                      ║')
    print('║   Submit:                                            ║')
    print('║     [1] report/main.pdf   (PDF)                      ║')
    print('║     [2] report/main.tex   (LaTeX source)             ║')
    print('║     [3] GitHub repo link                             ║')
    print('╚══════════════════════════════════════════════════════╝')
    print()

    if figs:
        print('  Generated figures:')
        for f in sorted(figs):
            print(f'    results/figures/{f}')
    print()


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rock Music SOC — Full Pipeline Runner')
    parser.add_argument('--quick', action='store_true',
        help='Fast test run (T=50,000 steps, ~1 min)')
    parser.add_argument('--sweep', action='store_true',
        help='Include connectivity sweep (adds time)')
    parser.add_argument('--analyze-only', action='store_true',
        help='Skip simulation, run analysis only')
    parser.add_argument('--T', type=int, default=500_000,
        help='Number of simulation steps (default: 500,000)')
    args = parser.parse_args()

    print_banner()
    check_dependencies()
    step_network()

    if not args.analyze_only:
        step_simulate(quick=args.quick,
                      sweep=args.sweep,
                      T=args.T)
    else:
        print('\n  [SKIP] Simulation skipped (--analyze-only)')

    step_analyze()
    print_summary()
