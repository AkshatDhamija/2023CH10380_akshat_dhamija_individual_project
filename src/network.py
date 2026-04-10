"""
network.py
----------
Constructs and analyses the rock music influence network
from the verified edge list in data/influence_edges.csv.

Provides:
    build_graph()          -> nx.DiGraph
    print_network_stats()  -> None
    get_thresholds()       -> dict {node: int}
    plot_network()         -> saves fig to results/figures/

Author : Akshat
Course : Complexity Science, IIT Delhi, 2026
"""

import os
import csv
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'influence_edges.csv')
FIG_DIR   = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Era → colour map (for visualisation) ───────────────────────
ERA_COLOURS = {
    '1930s': '#6B4226',   # dark brown  — Blues roots
    '1950s': '#E67E22',   # orange      — Early Rock & R&B
    '1960s': '#2980B9',   # blue        — British Invasion / Psychedelic
    '1970s': '#C0392B',   # red         — Hard Rock / Punk
    '1980s': '#8E44AD',   # purple      — Metal / Alternative
    '1990s': '#27AE60',   # green       — Grunge / Alt Rock
    '2000s': '#17A589',   # teal        — Post-Punk Revival
}


def _parse_era(era_string: str) -> str:
    """Extract the decade string (e.g. '1960s') from an era field."""
    for decade in ERA_COLOURS:
        if decade in era_string:
            return decade
    return 'Unknown'


# ══════════════════════════════════════════════════════════════
def build_graph(csv_path: str = DATA_PATH) -> nx.DiGraph:
    """
    Read influence_edges.csv and return a validated DiGraph.

    Each node gets attributes:
        era        : decade string  ('1960s', '1970s', …)
        out_degree : number of artists this node directly influenced

    Each edge gets attributes:
        genre_transition : e.g. 'Blues to Psychedelic'
        verified_source  : citation string
    """
    G = nx.DiGraph()

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row['source'].strip()
            tgt = row['target'].strip()

            # Add nodes with era metadata
            if src not in G:
                G.add_node(src, era=_parse_era(row['source_era']))
            if tgt not in G:
                G.add_node(tgt, era=_parse_era(row['target_era']))

            # Add edge with metadata
            G.add_edge(
                src, tgt,
                genre_transition = row['genre_transition'].strip(),
                verified_source  = row['verified_source'].strip(),
            )

    # Attach out-degree as node attribute
    for node in G.nodes():
        G.nodes[node]['out_degree'] = G.out_degree(node)

    return G


# ══════════════════════════════════════════════════════════════
def get_thresholds(G: nx.DiGraph, lam: float = 1.0) -> dict:
    """
    Return toppling thresholds for every node.

    threshold_v = max(1, floor(lam * out_degree_v))

    Leaf nodes (out_degree = 0) get threshold = 1 but
    will never topple because the avalanche rule skips them.
    lam is the connectivity scaling factor for the sweep.
    """
    return {
        v: max(1, int(lam * G.out_degree(v)))
        for v in G.nodes()
    }


# ══════════════════════════════════════════════════════════════
def print_network_stats(G: nx.DiGraph) -> None:
    """Print a formatted summary of the network."""
    nodes      = G.number_of_nodes()
    edges      = G.number_of_edges()
    out_degrees = [G.out_degree(v) for v in G.nodes()]
    in_degrees  = [G.in_degree(v)  for v in G.nodes()]

    # Hubs: top 5 by out-degree
    sorted_nodes = sorted(G.nodes(), key=lambda v: G.out_degree(v),
                          reverse=True)

    print("=" * 55)
    print("  Rock Music Influence Network — Statistics")
    print("=" * 55)
    print(f"  Nodes (artists)        : {nodes}")
    print(f"  Edges (influence links): {edges}")
    print(f"  Mean out-degree        : {np.mean(out_degrees):.2f}")
    print(f"  Max  out-degree        : {max(out_degrees)}"
          f"  ({sorted_nodes[0]})")
    print(f"  Mean in-degree         : {np.mean(in_degrees):.2f}")
    print(f"  Max  in-degree         : {max(in_degrees)}"
          f"  ({max(G.nodes(), key=lambda v: G.in_degree(v))})")
    print(f"  Leaf nodes (out-deg=0) : "
          f"{sum(1 for d in out_degrees if d == 0)}")

    # Weakly connected components
    wcc = list(nx.weakly_connected_components(G))
    print(f"  Weakly connected comps : {len(wcc)}")

    try:
        # Diameter on largest WCC
        largest = max(wcc, key=len)
        sub     = G.subgraph(largest).to_undirected()
        diam    = nx.diameter(sub)
        apl     = nx.average_shortest_path_length(sub)
        print(f"  Diameter (largest WCC) : {diam}")
        print(f"  Avg path length        : {apl:.2f}")
    except Exception:
        print(f"  Diameter               : (graph not connected)")

    print(f"\n  Top 8 hubs by out-degree:")
    for v in sorted_nodes[:8]:
        print(f"    {v:<30s}  out={G.out_degree(v)}"
              f"  in={G.in_degree(v)}")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════
def plot_network(G: nx.DiGraph, save: bool = True) -> None:
    """
    Draw the influence network with nodes coloured by era.
    Node size scaled by out-degree (hub prominence).
    Saves to results/figures/fig_network.png
    """
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    # Layout: hierarchical left-to-right by era decade
    era_order = ['1930s', '1950s', '1960s', '1970s',
                 '1980s', '1990s', '2000s', 'Unknown']
    era_x = {e: i * 2.2 for i, e in enumerate(era_order)}

    # Assign y positions within each era column
    era_nodes: dict = {e: [] for e in era_order}
    for v in G.nodes():
        era = G.nodes[v].get('era', 'Unknown')
        era_nodes[era].append(v)

    pos = {}
    rng = np.random.default_rng(99)   # fixed seed for reproducibility
    for era, nodes_in_era in era_nodes.items():
        x = era_x[era]
        n = len(nodes_in_era)
        ys = np.linspace(-n / 2, n / 2, max(n, 1))
        rng.shuffle(ys)
        for v, y in zip(nodes_in_era, ys):
            pos[v] = (x, y)

    # Node sizes and colours
    node_colours = [ERA_COLOURS.get(G.nodes[v].get('era', 'Unknown'),
                                    '#AAAAAA')
                    for v in G.nodes()]
    node_sizes   = [120 + 80 * G.out_degree(v) for v in G.nodes()]

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#AAAAAA', alpha=0.45,
        arrows=True, arrowsize=10,
        width=0.8,
        connectionstyle='arc3,rad=0.08',
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colours,
        node_size=node_sizes,
        alpha=0.92,
    )

    # Labels: only show artists with out-degree >= 2
    label_nodes = {v: v for v in G.nodes() if G.out_degree(v) >= 2}
    nx.draw_networkx_labels(
        G, pos, labels=label_nodes, ax=ax,
        font_size=7, font_weight='bold', font_color='#1A1A2E',
    )

    # Legend for eras
    patches = [
        mpatches.Patch(color=col, label=era)
        for era, col in ERA_COLOURS.items()
        if any(G.nodes[v].get('era') == era for v in G.nodes())
    ]
    ax.legend(handles=patches, title='Era', loc='lower right',
              fontsize=8, title_fontsize=9,
              framealpha=0.85, edgecolor='#CCCCCC')

    ax.set_title(
        'Rock Music Influence Network\n'
        'Node size ∝ out-degree  |  Colour = generational era',
        fontsize=13, fontweight='bold', color='#1A1A2E', pad=12,
    )
    ax.axis('off')
    plt.tight_layout()

    if save:
        path = os.path.join(FIG_DIR, 'fig_network.png')
        fig.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'[NET] Network figure saved -> {path}')

    plt.close(fig)


# ══════════════════════════════════════════════════════════════
def plot_degree_distribution(G: nx.DiGraph, save: bool = True) -> None:
    """
    Plot the out-degree distribution on log-log axes.
    A heavy tail here motivates the scale-free / SOC framework.
    """
    out_degs = [G.out_degree(v) for v in G.nodes() if G.out_degree(v) > 0]
    deg_counts = {}
    for d in out_degs:
        deg_counts[d] = deg_counts.get(d, 0) + 1

    ks = sorted(deg_counts.keys())
    ps = [deg_counts[k] / len(out_degs) for k in ks]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(ks, ps, color='#C0392B', s=55, zorder=5,
               label='Observed out-degree')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Out-degree  $k$  (number of artists influenced)',
                  fontsize=11)
    ax.set_ylabel('Probability  $P(k)$', fontsize=11)
    ax.set_title('Out-degree Distribution of the Influence Network\n'
                 'Heavy tail consistent with scale-free topology',
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(FIG_DIR, 'fig_degree_dist.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[NET] Degree distribution saved -> {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    G = build_graph()
    print_network_stats(G)
    plot_network(G)
    plot_degree_distribution(G)
