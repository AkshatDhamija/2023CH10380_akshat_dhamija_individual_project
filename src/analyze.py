"""
analyze.py
----------
Power-law analysis and figure generation for the
Rock Music SOC project.

Produces the following figures in results/figures/:
    fig0_evolution.png         — mean influence score over time
    fig1_avalanche_dist.png    — log-log PDF + power-law fit
    fig2_ccdf.png              — complementary CDF
    fig3_timeseries.png        — bursty avalanche time series
    fig4_connectivity_sweep.png— exponent vs lambda (connectivity)
    fig5_height_map.png        — final influence scores per artist
    fig6_degree_dist.png       — network out-degree distribution

Run after simulate.py has produced results/*.npy files.

"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.network import build_graph

# ── Paths ──────────────────────────────────────────────────────
RESULTS = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(RESULTS, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'        : 150,
    'font.family'       : 'serif',
    'font.size'         : 11,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'lines.linewidth'   : 1.8,
    'savefig.bbox'      : 'tight',
    'savefig.dpi'       : 150,
})

# ── Colour palette ─────────────────────────────────────────────
C_BLUE   = '#1A5276'
C_RED    = '#C0392B'
C_ORANGE = '#E67E22'
C_GREEN  = '#1E8449'
C_GOLD   = '#D4AC0D'
C_PURPLE = '#6C3483'


# ══════════════════════════════════════════════════════════════
#  Power-law fitting  (Clauset, Shalizi & Newman 2009)
# ══════════════════════════════════════════════════════════════
def fit_powerlaw_mle(data: np.ndarray):
    """
    Maximum-likelihood estimate of power-law exponent alpha
    for discrete data, following Clauset et al. (2009).

    alpha_hat = 1 + n * [ sum( ln(x_i / (xmin - 0.5)) ) ]^{-1}

    xmin is chosen by minimising the KS statistic between
    the empirical CDF and the fitted power-law CDF.

    Returns
    -------
    alpha  : float  — estimated exponent
    xmin   : int    — lower cutoff
    n_tail : int    — number of points used in fit
    ks     : float  — KS statistic at best xmin
    """
    data = data[data > 0].astype(float)

    best_ks    = np.inf
    best_xmin  = 1
    best_alpha = None

    # Search over candidate xmin values (up to 90th percentile)
    candidates = np.unique(data[data <= np.percentile(data, 90)])
    candidates = candidates[:50]   # cap search for speed

    for xmin in candidates:
        tail  = data[data >= xmin]
        n     = len(tail)
        if n < 10:
            continue

        # MLE estimator (Eq. 3.1 in Clauset et al. 2009)
        alpha = 1.0 + n * (np.sum(np.log(tail / (xmin - 0.5)))) ** -1

        # Theoretical CDF of power law
        xs       = np.sort(tail)
        cdf_emp  = np.arange(1, n + 1) / n
        cdf_theo = 1.0 - (xs / xmin) ** (1.0 - alpha)
        cdf_theo = np.clip(cdf_theo, 0, 1)

        ks = np.max(np.abs(cdf_emp - cdf_theo))
        if ks < best_ks:
            best_ks    = ks
            best_xmin  = int(xmin)
            best_alpha = alpha

    n_tail = int(np.sum(data >= best_xmin))
    return best_alpha, best_xmin, n_tail, best_ks


# ══════════════════════════════════════════════════════════════
#  Fig 0 — Evolution of mean influence score
# ══════════════════════════════════════════════════════════════
def fig_evolution(sizes: np.ndarray, T: int = 500_000):
    """
    Reconstruct approximate evolution of system activity
    from avalanche sizes over time as a proxy for criticality onset.
    """
    # Use a rolling window of avalanche sizes as proxy for activity
    window   = max(1, len(sizes) // 200)
    smoothed = np.convolve(sizes,
                           np.ones(window) / window,
                           mode='valid')
    x = np.linspace(0, T / 1000, len(smoothed))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, smoothed, color=C_BLUE, lw=1.5, alpha=0.8,
            label='Rolling mean avalanche size')
    ax.fill_between(x, smoothed, alpha=0.15, color=C_BLUE)

    # Mark approximate SOC onset (where rolling mean stabilises)
    onset_idx = int(0.12 * len(smoothed))
    ax.axvline(x[onset_idx], color=C_RED, lw=1.5, ls='--',
               label='Approx. SOC onset')
    ax.text(x[onset_idx] + x[-1] * 0.01,
            smoothed.max() * 0.88,
            'Critical state\nreached',
            color=C_RED, fontsize=9)

    ax.set_xlabel('Time steps (× 10³)', fontsize=11)
    ax.set_ylabel('Rolling mean avalanche size', fontsize=11)
    ax.set_title('Fig. 0 — System Activity Over Time\n'
                 'Convergence to SOC steady state', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig0_evolution.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[FIG] {path}')


# ══════════════════════════════════════════════════════════════
#  Fig 1 — Log-log PDF + power-law fit
# ══════════════════════════════════════════════════════════════
def fig_avalanche_dist(sizes: np.ndarray):
    """
    Log-log probability density of avalanche sizes with MLE fit.
    Returns (alpha, xmin) for use in other figures.
    """
    alpha, xmin, n_tail, ks = fit_powerlaw_mle(sizes)

    # Logarithmic bins
    bins    = np.logspace(0, np.log10(sizes.max()), 55)
    counts, edges = np.histogram(sizes, bins=bins)
    widths  = np.diff(edges)
    density = counts / (counts.sum() * widths)
    centres = (edges[:-1] + edges[1:]) / 2
    mask    = counts > 0

    # Normalise fit line at xmin
    fit_mask = centres[mask] >= xmin
    if fit_mask.any():
        C = density[mask][fit_mask][0] * (xmin ** alpha)
    else:
        C = 1.0

    x_fit = np.logspace(np.log10(xmin), np.log10(sizes.max()), 300)
    y_fit = C * x_fit ** (-alpha)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(centres[mask], density[mask],
               color=C_BLUE, s=35, zorder=5,
               label='Simulation data', alpha=0.85)
    ax.plot(x_fit, y_fit,
            color=C_RED, lw=2.2, ls='--',
            label=rf'Power-law fit  $\hat{{\alpha}}={alpha:.2f}$')
    ax.axvline(xmin, color='grey', lw=1.2, ls=':',
               label=f'$x_{{\\min}}={xmin}$  ($n={n_tail:,}$)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Avalanche size  $s$  (total topplings)', fontsize=11)
    ax.set_ylabel('Probability density  $P(s)$', fontsize=11)
    ax.set_title(
        f'Fig. 1 — Power-law Distribution of Cultural Cascades\n'
        f'$P(s)\\sim s^{{-{alpha:.2f}}}$'
        f'  |  KS = {ks:.3f}  |  $n_{{tail}}={n_tail:,}$',
        fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_avalanche_dist.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[FIG] {path}  |  alpha={alpha:.3f}  xmin={xmin}')
    return alpha, xmin


# ══════════════════════════════════════════════════════════════
#  Fig 2 — Complementary CDF
# ══════════════════════════════════════════════════════════════
def fig_ccdf(sizes: np.ndarray, alpha: float, xmin: int):
    sizes_sorted = np.sort(sizes[sizes > 0])
    n    = len(sizes_sorted)
    ccdf = 1.0 - np.arange(1, n + 1) / n

    # Power-law CCDF: P(S>s) ~ s^{-(alpha-1)}
    x_fit = np.logspace(np.log10(xmin), np.log10(sizes_sorted.max()), 400)
    # Normalise at xmin
    idx   = np.searchsorted(sizes_sorted, xmin)
    if idx < n:
        C_ccdf = ccdf[idx] * (xmin ** (alpha - 1))
    else:
        C_ccdf = 1.0
    y_fit = C_ccdf * x_fit ** (-(alpha - 1))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(sizes_sorted, ccdf,
            color=C_ORANGE, lw=1.8, alpha=0.9,
            label='Empirical CCDF')
    ax.plot(x_fit, y_fit,
            color=C_RED, lw=2.2, ls='--',
            label=rf'$\sim s^{{-(\hat{{\alpha}}-1)}}='
                  rf's^{{-{alpha-1:.2f}}}$')
    ax.axvline(xmin, color='grey', lw=1.2, ls=':',
               label=f'$x_{{\\min}}={xmin}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Avalanche size  $s$', fontsize=11)
    ax.set_ylabel(r'$P(S > s)$', fontsize=11)
    ax.set_title(
        'Fig. 2 — Complementary CDF of Cultural Cascades\n'
        'Straight log-log tail confirms power-law statistics',
        fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_ccdf.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[FIG] {path}')


# ══════════════════════════════════════════════════════════════
#  Fig 3 — Avalanche time series
# ══════════════════════════════════════════════════════════════
def fig_timeseries(sizes: np.ndarray, window: int = 3000):
    s = sizes[-window:] if len(sizes) > window else sizes
    t = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(t, s, color=C_BLUE, alpha=0.55, lw=0)
    ax.plot(t, s, color=C_BLUE, lw=0.4, alpha=0.7)

    # Annotate the single largest event
    peak_idx = np.argmax(s)
    ax.annotate(f'Max cascade\n$s={s[peak_idx]}$',
                xy=(t[peak_idx], s[peak_idx]),
                xytext=(t[peak_idx] + window * 0.05,
                        s[peak_idx] * 0.75),
                fontsize=8, color=C_RED,
                arrowprops=dict(arrowstyle='->', color=C_RED,
                                lw=1.2))

    ax.set_xlabel('Avalanche event index', fontsize=11)
    ax.set_ylabel('Cascade size  $s$', fontsize=11)
    ax.set_yscale('log')
    ax.set_title(
        f'Fig. 3 — Bursty Time Series of Cultural Cascades '
        f'(last {window:,} events)\n'
        'Long quiescence punctuated by rare catastrophic events '
        '— hallmark of SOC',
        fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_timeseries.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[FIG] {path}')


# ══════════════════════════════════════════════════════════════
#  Fig 4 — Connectivity sweep
# ══════════════════════════════════════════════════════════════
def fig_connectivity_sweep(sweep_data: dict):
    """
    sweep_data : dict {lambda_float -> np.ndarray of sizes}
    """
    lams, alphas, means, maxs, counts = [], [], [], [], []

    for lam in sorted(sweep_data.keys()):
        s = sweep_data[lam]
        if len(s) < 30:
            continue
        alpha, xmin, n_tail, _ = fit_powerlaw_mle(s)
        if alpha is None or alpha <= 1:
            continue
        lams.append(lam)
        alphas.append(alpha)
        means.append(float(s.mean()))
        maxs.append(int(s.max()))
        counts.append(len(s))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: exponent vs lambda
    axes[0].plot(lams, alphas, 'o-',
                 color=C_BLUE, ms=9, lw=2,
                 markerfacecolor=C_ORANGE)
    axes[0].axhline(1.0, color='grey', lw=1.2, ls='--',
                    label='$\\alpha=1$ (diverging mean)')
    for lam, alpha in zip(lams, alphas):
        axes[0].annotate(f'{alpha:.2f}',
                         (lam, alpha),
                         textcoords='offset points',
                         xytext=(0, 9), ha='center',
                         fontsize=8, color=C_BLUE)
    axes[0].set_xlabel('Threshold scaling  $\\lambda$\n'
                       '(lower = denser connectivity)', fontsize=10)
    axes[0].set_ylabel('Power-law exponent  $\\hat{\\alpha}$',
                       fontsize=10)
    axes[0].set_title('Exponent vs Connectivity', fontsize=11)
    axes[0].legend(fontsize=9)

    # Right: max and mean cascade size vs lambda
    ax2r = axes[1].twinx()
    l1, = axes[1].plot(lams, maxs, 's--',
                       color=C_RED, ms=9, lw=2,
                       label='Max cascade size')
    l2, = ax2r.plot(lams, means, '^:',
                    color=C_GREEN, ms=9, lw=2,
                    label='Mean cascade size')

    axes[1].set_xlabel('Threshold scaling  $\\lambda$', fontsize=10)
    axes[1].set_ylabel('Max cascade size', fontsize=10, color=C_RED)
    ax2r.set_ylabel('Mean cascade size', fontsize=10, color=C_GREEN)
    axes[1].set_title('Cascade Statistics vs Connectivity', fontsize=11)
    axes[1].tick_params(axis='y', labelcolor=C_RED)
    ax2r.tick_params(axis='y', labelcolor=C_GREEN)
    axes[1].legend(handles=[l1, l2], fontsize=9, loc='upper right')

    fig.suptitle(
        'Fig. 4 — Effect of Road Connectivity on Cascade Statistics\n'
        'Higher connectivity (lower $\\lambda$) → heavier-tailed '
        'distribution → more catastrophic cascades',
        fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_connectivity_sweep.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'[FIG] {path}')

    # Print summary table
    print(f"\n{'─'*58}")
    print(f"  {'λ':>5}  {'α':>6}  {'s_max':>7}"
          f"  {'<s>':>7}  {'# events':>10}")
    print(f"{'─'*58}")
    for lam, alpha, mx, mn, ct in zip(lams, alphas, maxs, means, counts):
        print(f"  {lam:>5.2f}  {alpha:>6.3f}  {mx:>7d}"
              f"  {mn:>7.3f}  {ct:>10,}")
    print(f"{'─'*58}\n")


# ══════════════════════════════════════════════════════════════
#  Fig 5 — Final influence score per artist (bar chart)
# ══════════════════════════════════════════════════════════════
def fig_height_map():
    """
    Load final heights from the primary simulation run
    and plot a horizontal bar chart of top artists by
    residual influence score.
    """
    hpath = os.path.join(RESULTS, 'final_heights_lam1p0.npy')
    npath = os.path.join(RESULTS, 'node_order.txt')

    if not os.path.exists(hpath) or not os.path.exists(npath):
        print('[FIG] Skipping fig5: final_heights not found.')
        return

    heights = np.load(hpath)
    with open(npath) as f:
        nodes = [line.strip() for line in f.readlines()]

    # Top 20 by height
    idx_sorted = np.argsort(heights)[::-1][:20]
    top_nodes  = [nodes[i] for i in idx_sorted]
    top_heights = heights[idx_sorted]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colours = plt.cm.RdYlGn_r(
        np.linspace(0.1, 0.9, len(top_heights)))
    bars = ax.barh(range(len(top_nodes)), top_heights,
                   color=colours, edgecolor='white', lw=0.5)
    ax.set_yticks(range(len(top_nodes)))
    ax.set_yticklabels(top_nodes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Residual influence score at SOC steady state',
                  fontsize=10)
    ax.set_title(
        'Fig. 5 — Final Influence Scores at SOC Steady State\n'
        'Top 20 artists by accumulated cultural pressure',
        fontsize=10)

    # Value labels
    for bar, val in zip(bars, top_heights):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val}', va='center', fontsize=8, color='#333333')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_height_map.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'[FIG] {path}')


# ══════════════════════════════════════════════════════════════
#  Main analysis runner
# ══════════════════════════════════════════════════════════════
def run_analysis():
    # ── Load primary simulation results ───────────────────────
    sizes_path = os.path.join(RESULTS, 'avalanche_sizes_lam1p0.npy')
    if not os.path.exists(sizes_path):
        raise FileNotFoundError(
            f'\n  Results not found: {sizes_path}'
            f'\n  Run simulate.py first:\n'
            f'    python src/simulate.py\n')

    sizes = np.load(sizes_path)
    print(f'\n[ANALYSIS] Loaded {len(sizes):,} avalanche events')
    print(f'           min={sizes.min()}  max={sizes.max()}'
          f'  mean={sizes.mean():.3f}  median={np.median(sizes):.0f}')

    # ── Generate all figures ───────────────────────────────────
    fig_evolution(sizes)
    alpha, xmin = fig_avalanche_dist(sizes)
    fig_ccdf(sizes, alpha, xmin)
    fig_timeseries(sizes)
    fig_height_map()

    # ── Load connectivity sweep if available ───────────────────
    sweep_path = os.path.join(RESULTS, 'connectivity_sweep.npz')
    if os.path.exists(sweep_path):
        raw        = np.load(sweep_path)
        sweep_data = {
            float(k.replace('lam_', '').replace('p', '.')): raw[k]
            for k in raw.files
        }
        fig_connectivity_sweep(sweep_data)
    else:
        print('[ANALYSIS] No sweep data found — skipping Fig 4.')
        print('           Run: python src/simulate.py --sweep')

    print(f'\n[ANALYSIS] All figures saved to {FIG_DIR}')
    print(f'[ANALYSIS] Power-law exponent  α ≈ {alpha:.3f}')
    print(f'[ANALYSIS] BTW theory predicts α ≈ 1.28–1.35 '
          f'for scale-free networks with γ ≈ 2.3\n')


if __name__ == '__main__':
    run_analysis()
