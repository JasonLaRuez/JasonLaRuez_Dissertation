"""
Module Name: LittkeSIS.py
Description: Contains functions for loading, preprocessing
processing, and topologically analyzing the LittleSIS relation
database from littlesis.org

Author: Jason LaRuez
Date: 2026
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import random
from tqdm.notebook import tqdm # Allows for real-time progress bar of simulations
import gc
from itertools import combinations
import time
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from collections import Counter

import dionysus as d

# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

def missing_value_audit(data_dict, dict_name, attr_key='attributes'):
    """
    Computes, for each attribute across a dictionary of entities or
    relationships, the proportion of entries where that attribute is
    missing (None, empty string, empty list, or empty dict), and
    prints a formatted summary table.

    Parameters
    ----------
    data_dict : dict
      Entities or relationships, keyed by id.
    dict_name : str
      Name used in the printed summary header.
    attr_key : str or None
      Key within each entry that holds its attributes dict. Pass None
      if entries are already flat attribute dicts (e.g. after
      preprocessing).

    Returns
    -------
    df : pandas.DataFrame
      One row per attribute, with columns 'attribute', 'total',
      'n_missing', 'n_present', 'pct_missing', 'pct_present', sorted
      by pct_missing descending.
    """
    from collections import defaultdict

    counts_missing = defaultdict(int)
    counts_total   = defaultdict(int)
    all_keys       = set()

    for entry_id, entry in data_dict.items():
        # Get the attributes dict
        if attr_key is not None:
            attrs = entry.get(attr_key, {}) if isinstance(entry, dict) else {}
        else:
            attrs = entry if isinstance(entry, dict) else {}

        all_keys.update(attrs.keys())

    # Second pass: count missing per key
    for entry_id, entry in data_dict.items():
        if attr_key is not None:
            attrs = entry.get(attr_key, {}) if isinstance(entry, dict) else {}
        else:
            attrs = entry if isinstance(entry, dict) else {}

        for key in all_keys:
            counts_total[key] += 1
            val = attrs.get(key, None)
            if val is None:
                counts_missing[key] += 1
            elif isinstance(val, str) and val.strip() == '':
                counts_missing[key] += 1
            elif isinstance(val, (list, dict)) and len(val) == 0:
                counts_missing[key] += 1

    rows = []
    for key in all_keys:
        total   = counts_total[key]
        missing = counts_missing[key]
        rows.append({
            'attribute':        key,
            'total':            total,
            'n_missing':        missing,
            'n_present':        total - missing,
            'pct_missing':      100 * missing / total if total > 0 else np.nan,
            'pct_present':      100 * (total - missing) / total if total > 0 else np.nan,
        })

    df = pd.DataFrame(rows).sort_values('pct_missing', ascending=False)
    df = df.reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Missing value audit: {dict_name} (n={len(data_dict):,})")
    print(f"{'='*60}")
    print(df[['attribute', 'n_present', 'n_missing',
              'pct_missing', 'pct_present']].to_string(
        index=False,
        formatters={
            'pct_missing': lambda x: f'{x:6.2f}%',
            'pct_present': lambda x: f'{x:6.2f}%',
            'n_missing':   lambda x: f'{int(x):>10,}',
            'n_present':   lambda x: f'{int(x):>10,}',
        }
    ))
    return df


def plot_missing_audit(df, title, save_path=None):
    """
    Plots missing-value proportions (see missing_value_audit) as a
    horizontal bar chart, sorted from least to most missing.

    Parameters
    ----------
    df : pandas.DataFrame
      Output of missing_value_audit, with 'attribute' and
      'pct_missing' columns.
    title : str
      Plot title.
    save_path : str or None
      If given, the figure is saved to this path.

    Returns
    -------
    None
      Displays the figure; saves it to save_path if given.
    """
    df_plot = df.sort_values('pct_missing', ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_plot) * 0.35)),
                            constrained_layout=True)

    bars = ax.barh(df_plot['attribute'], df_plot['pct_missing'],
                   color='steelblue', edgecolor='none')

    # Annotate bars with percentage
    for bar, pct in zip(bars, df_plot['pct_missing']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', va='center', ha='left', fontsize=8)

    ax.set_xlabel('% Missing or Blank', fontsize=11, labelpad=6)
    ax.set_xlim(0, 115)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def year_counts(years, bins):
    """
    Counts occurrences of each year, aligned to a set of bin edges.

    Parameters
    ----------
    years : iterable of int
      Year values to count.
    bins : sequence of int
      Bin edges; counts are reported for years bins[0]..bins[-2]
      (the left edge of each bin except the last).

    Returns
    -------
    ndarray
      Count of years equal to each of bins[:-1], in order.
    """
    c = Counter(years)
    return np.array([c.get(y, 0) for y in bins[:-1]])

def extract_update_years(rel_dict):
    """
    Extracts the year component of each relationship's 'updated_at'
    date, skipping relationships where it is missing.

    Parameters
    ----------
    rel_dict : dict
      LittleSIS relationship dicts, keyed by relationship id.

    Returns
    -------
    list of int
      One year per relationship with a non-null 'updated_at'.
    """
    return [int(rel_dict[r]['updated_at'][:4]) for r in rel_dict
            if rel_dict[r]['updated_at'] is not None]

def extract_start_years(rel_dict):
    """
    Extracts the year component of each relationship's 'start_date'.

    Parameters
    ----------
    rel_dict : dict
      LittleSIS relationship dicts, keyed by relationship id. Every
      relationship is assumed to have a 'start_date'.

    Returns
    -------
    list of int
      One year per relationship.
    """
    return [int(rel_dict[r]['start_date'][:4]) for r in rel_dict]

def extract_end_years(rel_dict):
    """
    Extracts the year component of each relationship's 'end_date',
    skipping relationships where it is missing (still ongoing).

    Parameters
    ----------
    rel_dict : dict
      LittleSIS relationship dicts, keyed by relationship id.

    Returns
    -------
    list of int
      One year per relationship with a non-null 'end_date'.
    """
    return [int(rel_dict[r]['end_date'][:4]) for r in rel_dict
            if rel_dict[r]['end_date'] is not None]

def extract_durations(rel_dict):
    """
    Computes the duration, in whole years, of each relationship that
    has both a start and end date, discarding negative durations.

    Parameters
    ----------
    rel_dict : dict
      LittleSIS relationship dicts, keyed by relationship id.

    Returns
    -------
    durations : list of int
      Non-negative year durations, one per relationship with both
      dates present.
    """
    durations = []
    for r in rel_dict:
        s = rel_dict[r]['start_date']
        e = rel_dict[r]['end_date']
        if s is not None and e is not None:
            dur = int(e[:4]) - int(s[:4])
            if dur >= 0:
                durations.append(dur)
    return durations

def GraphStats_LittleSIS(rel_dicts, sort_by='start_date', timing=False,
                          random_seed=42):
    """
    Builds a graph from LittleSIS relationship dicts by adding edges in
    date order (ties broken randomly). Computes mean degree, max
    degree, and average clustering coefficient at the end of each
    calendar year.

    Parameters
    ----------
    rel_dicts : list of dict
      Relationship dictionaries to draw edges from.
    sort_by : str
      Date field used to order edge additions: 'start_date' or
      'updated_at'.
    timing : bool
      If True, print progress messages and timing at each stage.
    random_seed : int
      Seed for tie-breaking among edges sharing the same date.

    Returns
    -------
    yearly_stats : dict
      Keys 'year', 'mean_degree', 'max_degree', 'avg_clustering', each
      a list indexed by (the index of) calendar year with at least
      one edge added.
    D : dict
      Terminal degree distribution, {node: degree}.
    G : networkx.Graph
      Terminal graph.
    """
    rng = random.Random(random_seed)

    # -------------------------------------------------------------------------
    # (1/3) Collect, sort, deduplicate edges
    # -------------------------------------------------------------------------
    if timing:
        print("(1/3) Collecting and sorting edges", flush=True)
        t0 = time.time()

    raw = []
    for rel_dict in rel_dicts:
        for rel in rel_dict.values():
            date_str = rel.get(sort_by)
            if not date_str:
                continue
            u, v = rel['entity1_id'], rel['entity2_id']
            if u == v:
                continue
            raw.append((date_str, rng.random(), u, v))

    raw.sort(key=lambda x: (x[0], x[1]))

    seen_pairs = set()
    filtered = []
    for date_str, _, u, v in raw:
        key = (min(u, v), max(u, v))
        if key not in seen_pairs:
            seen_pairs.add(key)
            filtered.append((date_str, u, v))

    if timing:
        print(f"  {len(filtered)} unique edges. "
              f"Time: {time.time()-t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # (2/3) Evolve graph, compute stats at end of each year
    # -------------------------------------------------------------------------
    if timing:
        print("(2/3) Evolving graph and computing yearly stats", flush=True)
        t0 = time.time()

    G   = nx.Graph()
    D   = {}   # degree distribution: D[node] = degree

    yearly_stats = {
        'year':            [],
        'mean_degree':     [],
        'max_degree':      [],
        'avg_clustering':  [],
    }

    # Group edges by year for efficient yearly processing
    edges_by_year = {}
    for date_str, u, v in filtered:
        yr = int(date_str[:4])
        if yr not in edges_by_year:
            edges_by_year[yr] = []
        edges_by_year[yr].append((u, v))

    all_years = sorted(edges_by_year.keys())

    for yr in tqdm(all_years) if timing else all_years:
        for u, v in edges_by_year[yr]:
            # Add nodes if new
            for node in [u, v]:
                if node not in G:
                    G.add_node(node)
                    D[node] = 0
            # Add edge and update degree distribution
            G.add_edge(u, v)
            D[u] += 1
            D[v] += 1

        # End-of-year snapshot
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            continue

        degrees      = list(D.values())
        mean_deg     = np.mean(degrees)
        max_deg      = np.max(degrees)
        avg_clust    = nx.average_clustering(G)

        yearly_stats['year'].append(yr)
        yearly_stats['mean_degree'].append(mean_deg)
        yearly_stats['max_degree'].append(max_deg)
        yearly_stats['avg_clustering'].append(avg_clust)

    if timing:
        print(f"  Done. Time: {time.time()-t0:.2f}s", flush=True)

    return yearly_stats, D, G


# =============================================================================
# Plot yearly graph statistics
# =============================================================================
def plot_graph_stats(yearly_stats, title='Graph Statistics — Annual Snapshots', save_path='graph_stats_yearly.pdf'):
    """
    Plots mean degree, max degree, and average clustering coefficient
    (see GraphStats_LittleSIS) as annual time series, side by side.

    Parameters
    ----------
    yearly_stats : dict
      Output of GraphStats_LittleSIS, with keys 'year', 'mean_degree',
      'max_degree', 'avg_clustering'.
    title : str
      Figure title.
    save_path : str
      Path the figure is saved to.

    Returns
    -------
    None
      Displays the figure and saves it to save_path.
    """
    years   = np.array(yearly_stats['year'])
    metrics = {
        'Mean Degree':                  yearly_stats['mean_degree'],
        'Max Degree':                   yearly_stats['max_degree'],
        'Avg. Clustering Coefficient':  yearly_stats['avg_clustering'],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, (label, vals) in zip(axes, metrics.items()):
        ax.plot(years, vals, color='steelblue', linewidth=1.2,
                marker='o', markersize=4, markerfacecolor='white',
                markeredgecolor='steelblue', markeredgewidth=1.0)
        ax.set_title(label, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel('Year', fontsize=10, labelpad=4)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{x:,.3g}')
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(title,
                 fontsize=13, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# Plot terminal degree distribution on log-log axes
# =============================================================================
def plot_degree_distribution(D, title='Terminal Degree Distribution', save_path='degree_distribution.pdf'):
    """
    Plots the terminal degree distribution on log-log axes (raw and
    log-binned), estimating and overlaying a power-law exponent via
    both OLS regression on log-binned density and MLE (via the
    powerlaw package) on the raw degree sequence.

    Parameters
    ----------
    D : dict
      Terminal degree distribution, {node: degree} (e.g. from
      GraphStats_LittleSIS).
    title : str
      Figure title.
    save_path : str
      Path the figure is saved to.

    Returns
    -------
    None
      Displays the figure, saves it to save_path, and prints the
      fitted OLS and MLE exponents.
    """
    from collections import Counter
    from scipy import stats
    import powerlaw

    deg_counts = Counter(D.values())
    degrees    = np.array(sorted(deg_counts.keys()))
    counts     = np.array([deg_counts[d] for d in degrees])
    probs      = counts / counts.sum()

    # Log-binned version
    log_bins    = np.logspace(np.log10(max(1, degrees.min())),
                               np.log10(degrees.max()), 30)
    bin_counts, bin_edges = np.histogram(list(D.values()), bins=log_bins)
    bin_widths  = np.diff(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_density = bin_counts / (bin_counts.sum() * bin_widths)
    nonzero     = bin_counts > 0

    # --- OLS fit on log-binned data (middle 80% of log-x range) ---
    log_x = np.log10(bin_centers[nonzero])
    log_y = np.log10(bin_density[nonzero])
    lo    = np.percentile(log_x, 10)
    hi    = np.percentile(log_x, 90)
    mask  = (log_x >= lo) & (log_x <= hi) & np.isfinite(log_y)
    slope, intercept, r, _, _ = stats.linregress(log_x[mask], log_y[mask])
    gamma_ols = -slope
    r2_ols    = r**2

    x_fit = np.logspace(np.log10(bin_centers[nonzero].min()),
                         np.log10(bin_centers[nonzero].max()), 200)
    y_fit = 10**intercept * x_fit**(-gamma_ols)

    # --- MLE fit via powerlaw package on raw degree sequence ---
    degree_sequence = list(D.values())
    fit = powerlaw.Fit(degree_sequence, discrete=True, verbose=False)
    gamma_mle = fit.power_law.alpha
    xmin_mle  = fit.power_law.xmin

    # MLE fit line anchored to middle log-bin
    mid_idx    = len(bin_centers[nonzero]) // 2
    x_mid      = bin_centers[nonzero][mid_idx]
    y_mid      = bin_density[nonzero][mid_idx]
    x_mle      = np.logspace(np.log10(xmin_mle),
                              np.log10(bin_centers[nonzero].max()), 200)
    norm_mle   = y_mid * (x_mid ** gamma_mle)
    y_mle      = norm_mle * x_mle**(-gamma_mle)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # --- Left: raw, no fit ---
    axes[0].scatter(degrees, probs, color='steelblue', s=12,
                    alpha=0.7, edgecolors='none', label='Data')
    axes[0].plot(x_mle, y_mle, color='darkorange', linewidth=1.5,
                 linestyle=':',
                 label=(rf'MLE: $\gamma = {gamma_mle:.2f}$'))
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Degree $k$', fontsize=16, labelpad=4)
    axes[0].set_ylabel('$P(k)$', fontsize=16, labelpad=4)
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].set_title('Degree Distribution', fontsize=16,
                       fontweight='bold', pad=6)
    axes[0].legend(fontsize=14, framealpha=0.9, edgecolor='#cccccc')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    axes[0].set_axisbelow(True)

    # --- Right: log-binned with OLS and MLE fits ---
    axes[1].scatter(bin_centers[nonzero], bin_density[nonzero],
                    color='steelblue', s=20, alpha=0.8,
                    edgecolors='none', label='Data')
    axes[1].plot(x_fit, y_fit, color='crimson', linewidth=1.5,
                 linestyle='--',
                 label=(rf'OLS: $\gamma = {gamma_ols:.2f}$, '
                        rf'$R^2 = {r2_ols:.3f}$'))
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Degree $k$', fontsize=16, labelpad=4)
    axes[1].set_ylabel('$p(k)$ (density)', fontsize=16, labelpad=4)
    axes[1].set_title('Degree Distribution (log-binned)', fontsize=16,
                       fontweight='bold', pad=6)
    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].legend(fontsize=14, framealpha=0.9, edgecolor='#cccccc')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    axes[1].set_axisbelow(True)

    fig.suptitle(title, fontsize=18,
                 fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"OLS (log-binned): gamma = {gamma_ols:.3f}, R² = {r2_ols:.3f}")
    print(f"MLE (raw sequence): gamma = {gamma_mle:.3f}, "
          f"xmin = {xmin_mle:.1f}")

def PH_LittleSIS(rel_dicts, sort_by='start_date', timing=False,
                 maxDim=np.inf, random_seed=42):
    """
    Builds a graph from LittleSIS relationship dicts by adding edges in
    date order (ties broken randomly with fixed seed), tracking the
    clique complex's simplex counts and cycle rank CR_1 (= N_1 - N_0 +
    beta_0) at each step, then computes persistent homology from the
    resulting filtration.

    Parameters
    ----------
    rel_dicts : list of dict
        Relationship dictionaries to draw edges from,
        e.g. [has_start_p2p, has_start_end_p2p].
        Pass multiple types (P-P, P-O, O-O) to build a multilayer graph.
    sort_by : str
        Date field used to order edge additions: 'start_date' or 'updated_at'.
    timing : bool
        If True, print progress messages and timing at each stage.
    maxDim : int or np.inf
        Maximum simplex dimension to track (e.g. maxDim=2 tracks up to triangles).
    random_seed : int
        Seed for tie-breaking among edges sharing the same date.

    Returns
    -------
    Betti        : np.ndarray, shape (4, n_steps)
    SimplexCounts: list of lists, one entry per edge-addition step
    Euler        : np.ndarray, shape (n_steps,)
    cycleR       : list, length n_steps; CR_1 (= N_1 - N_0 + beta_0) at
                   each step
    dates        : list of str, the sort_by date string for each step
    """
    rng = random.Random(random_seed)

    # -------------------------------------------------------------------------
    # (1/5) Collect, sort, and deduplicate edges
    # -------------------------------------------------------------------------
    if timing:
        print("(1/5) Collecting and sorting edges", flush=True)
        t0 = time.time()

    raw = []
    for rel_dict in rel_dicts:
        for rel in rel_dict.values():
            date_str = rel.get(sort_by)
            if not date_str:
                continue
            u, v = rel['entity1_id'], rel['entity2_id']
            if u == v:
                continue  # skip self-loops
            raw.append((date_str, rng.random(), u, v))

    # Primary sort by date, secondary sort by random key (tie-breaking)
    raw.sort(key=lambda x: (x[0], x[1]))

    # Keep only the first occurrence of each undirected pair
    seen_pairs = set()
    filtered = []
    for date_str, _, u, v in raw:
        key = (min(u, v), max(u, v))
        if key not in seen_pairs:
            seen_pairs.add(key)
            filtered.append((date_str, u, v))

    n_steps = len(filtered)
    dates = [e[0] for e in filtered]

    if timing:
        print(f"  {n_steps} unique edges. Time: {time.time()-t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # (2/5) Initialize graph and data structures
    # -------------------------------------------------------------------------
    if timing:
        print("(2/5) Initializing graph and data structures", flush=True)
        t0 = time.time()

    G = nx.Graph()
    node_seen   = set()
    Times       = []       # (simplex, filtration_time) pairs for dionysus
    SimplexCounts = []     # one list of counts per step
    cycleR      = np.zeros(n_steps)

    if timing:
        print(f"  Done. Time: {time.time()-t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # (3/5) Evolve graph edge by edge
    # -------------------------------------------------------------------------
    if timing:
        print("(3/5) Evolving graph", flush=True)
        t0 = time.time()

    iterator = enumerate(tqdm(filtered)) if timing else enumerate(filtered)
    for timer, (date_str, u, v) in iterator:

        # Carry forward simplex counts from previous step
        current_counts = SimplexCounts[-1].copy() if SimplexCounts else []
        new_simplices = set()

        # Track newly introduced nodes as 0-simplices
        for node in [u, v]:
            if node not in node_seen:
                node_seen.add(node)
                G.add_node(node)
                new_simplices.add((node,))

        # Common neighbors before edge addition — these are exactly the nodes
        # that will form new triangles (and higher simplices) with (u,v)
        common = list(nx.common_neighbors(G, u, v))

        # Add edge; track as 1-simplex
        G.add_edge(u, v)
        new_simplices.add(tuple(sorted([u, v])))

        # Find all new higher-order simplices. A simplex is new iff it contains
        # both u and v (since the edge u-v is what makes it possible).
        # We search cliques in the subgraph induced by {u, v} + common neighbors.
        if common:
            H = G.subgraph([u, v] + common)
            for clique in nx.find_cliques(H):
              if u in clique and v in clique:
                clique_s = sorted(clique)
                for size in range(3, min(len(clique_s), maxDim) + 1):
                    for face in combinations(clique_s, size):
                        if u in face and v in face:
                            new_simplices.add(face)

        # Update simplex counts and filtration record
        for simplex in new_simplices:
            dim = len(simplex) - 1
            if dim > maxDim:
                continue
            while len(current_counts) <= dim:
                current_counts.append(0)
            current_counts[dim] += 1
            if dim <= 3:  # dionysus only needs simplices up to dim 3 for H0-H3
                Times.append((list(simplex), timer))

        SimplexCounts.append(current_counts)

        # Cycle rank: |E| - |V| (beta_0 correction added after PH)
        cycleR[timer] = G.number_of_edges() - G.number_of_nodes()

    if timing:
        print(f"  Evolution complete. Time: {time.time()-t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # (4/5) Persistent Homology
    # -------------------------------------------------------------------------
    if timing:
        print("(4/5) Computing Persistent Homology", flush=True)
        t0 = time.time()

    f = d.Filtration(Times)
    del Times
    gc.collect()
    m = d.homology_persistence(f)

    if timing:
        print(f"  PH complete. Time: {time.time()-t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # (5/5) Extract Betti numbers and Euler characteristic
    # -------------------------------------------------------------------------
    if timing:
        print("(5/5) Extracting Betti numbers and Euler characteristic", flush=True)
        t0 = time.time()

    Betti = np.zeros((4, n_steps))
    dgms = d.init_diagrams(m, f)
    for i, dgm in enumerate(dgms):
        if i >= 4:
            break
        for p in dgm:
            birth = int(p.birth)
            death = int(min(p.death, n_steps))
            Betti[i][birth:death] += 1

    Euler = np.zeros(n_steps)
    for i, counts in enumerate(SimplexCounts):
        for j, c in enumerate(counts):
            Euler[i] += (-1)**j * c

    # Add beta_0 correction to cycle rank
    cycleR = [cycleR[i] + Betti[0][i] for i in range(n_steps)]

    if timing:
        print(f"  Done. Time: {time.time()-t0:.2f}s", flush=True)

    return Betti, SimplexCounts, Euler, cycleR, dates

def _sc(SimplexCounts, dim, n):
    """
    Extracts one dimension's simplex-count time series from a
    SimplexCounts result, which per CLAUDE.md is a list of lists
    (possibly ragged across timesteps), NOT a 2D numpy array.

    Parameters
    ----------
    SimplexCounts : list of list of float
      SimplexCounts[t] is the row of simplex counts by size at
      timestep t.
    dim : int
      Column index (simplex size - 1) to extract.
    n : int
      Number of timesteps to extract (0..n-1).

    Returns
    -------
    ndarray
      1-D array of length n, with 0 where a row is too short to have
      that column.
    """
    return np.array([
        SimplexCounts[t][dim] if len(SimplexCounts[t]) > dim else 0
        for t in range(n)
    ], dtype=float)

def _aggregate_yearly(arr, years, unique_years):
    """
    Reduces a per-step time series to one end-of-year snapshot per
    year, taking each year's last observed value.

    Parameters
    ----------
    arr : ndarray
      Per-step values.
    years : ndarray
      Calendar year of each step in arr, aligned by index.
    unique_years : ndarray
      Years to report a snapshot for, in order.

    Returns
    -------
    ndarray
      One value per entry in unique_years: the last arr value in that
      year, or NaN if the year has no steps.
    """
    out = np.full(len(unique_years), np.nan)
    for i, yr in enumerate(unique_years):
        mask = years == yr
        if np.any(mask):
            out[i] = arr[mask][-1]
    return out

def _style_ax(ax, xs, y, xlabel, x_is_dates=False):
    """
    Plots one line series with this module's standard marker/line
    style and axis formatting (rotated x tick labels, compact y tick
    formatting, and either yearly or date-based x ticks).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
      Axis to draw on.
    xs : array-like
      X values.
    y : array-like
      Y values.
    xlabel : str
      X-axis label.
    x_is_dates : bool
      If True, format the x-axis for datetime values (4-year ticks);
      otherwise use integer-valued ticks.

    Returns
    -------
    None
      Draws onto ax and sets its axis formatting.
    """
    LINE_COLOR = '#2c6fad'
    LINE_KW    = dict(color=LINE_COLOR, lw=1.4, marker='o', markersize=3.5,
                  markerfacecolor='white', markeredgecolor=LINE_COLOR,
                  markeredgewidth=1.0)
    ax.plot(xs, y, **LINE_KW)
    if x_is_dates:
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.3g}'))

def _yearly_xs_and_series(Betti, SimplexCounts, Euler, cycleR, dates):
    """
    Computes normalized Betti numbers, cycle ranks, and filling
    efficiencies from a persistent-homology run, and aggregates them
    (along with simplex counts, Betti numbers, and Euler
    characteristic) to end-of-year snapshots.

    Derived quantities, per CLAUDE.md definitions: tb0/tb1/tb2 are the
    normalized Betti numbers beta_tilde_k = beta_k / N_k; cr (input
    cycleR) is CR_1 = N_1 - N_0 + beta_0; cr1b1 = CR_1 - beta_1 (the
    number of independent 1-cycles filled by 2-simplices); cr2 is
    CR_2 = N_2 - CR_1 + beta_1 (the same construction one dimension
    up); cr2b2 = CR_2 - beta_2; g1 is gamma_1 = (CR_1 - beta_1) / N_2
    (the fraction of ADDED 2-simplices that filled a 1-cycle, not the
    fraction of 1-cycles filled); g2 is gamma_2 = (CR_2 - beta_2) / N_3,
    the analogous fraction one dimension up.

    Parameters
    ----------
    Betti : ndarray, shape (4, n_steps)
      Betti[k][t] is the k-th Betti number at step t.
    SimplexCounts : list of list of float
      Simplex counts by size at each step (list of lists, per
      CLAUDE.md convention).
    Euler : ndarray, shape (n_steps,)
      Euler characteristic at each step.
    cycleR : array-like, length n_steps
      CR_1 at each step.
    dates : list of str
      Date string (year in the first 4 characters) for each step.

    Returns
    -------
    unique_years : ndarray
      Calendar years spanned, one entry per year with a snapshot.
    yearly : dict
      Maps each of 's0','s1','s2','b0','b1','b2','tb0','tb1','tb2',
      'cr1b1','g1','cr2b2','g2','chi' to its end-of-year-aggregated
      ndarray (see _aggregate_yearly), aligned with unique_years.
    """
    n    = len(SimplexCounts)
    b0   = Betti[0].astype(float)
    b1   = Betti[1].astype(float)
    b2   = Betti[2].astype(float)
    s0   = _sc(SimplexCounts, 0, n)
    s1   = _sc(SimplexCounts, 1, n)
    s2   = _sc(SimplexCounts, 2, n)
    s3   = _sc(SimplexCounts, 3, n)
    cr   = np.array(cycleR, dtype=float)
    chi  = np.array(Euler,  dtype=float)

    with np.errstate(invalid='ignore', divide='ignore'):
        tb0    = np.where(s0 > 0, b0 / s0, np.nan)
        tb1    = np.where(s1 > 0, b1 / s1, np.nan)
        tb2    = np.where(s2 > 0, b2 / s2, np.nan)
        cr1b1  = cr - b1
        cr2    = s2 - cr + b1            # CR2 = S2 - CR1 + b1
        cr2b2  = cr2 - b2
        g1     = np.where(s2 > 0, cr1b1 / s2, np.nan)
        g2     = np.where(s3 > 0, cr2b2 / s3, np.nan)

    years        = np.array([int(d[:4]) for d in dates])
    unique_years = np.arange(years.min(), years.max() + 1)

    def agg(arr):
        return _aggregate_yearly(arr, years, unique_years)

    raw = {
        's0': s0, 's1': s1, 's2': s2,
        'b0': b0, 'b1': b1, 'b2': b2,
        'tb0': tb0, 'tb1': tb1, 'tb2': tb2,
        'cr1b1': cr1b1, 'g1': g1,
        'cr2b2': cr2b2, 'g2': g2,
        'chi': chi,
    }
    yearly = {k: agg(v) for k, v in raw.items()}
    return unique_years, yearly

# ── Figure 1: N0 N1 N2 | b0 b1 b2 | tb0 tb1 tb2 ─────────────────────────────

def plot_ph_simplices_betti(Betti, SimplexCounts, Euler, cycleR, dates,
                            title='', save_path=None):
    """
    Plots a 3x3 figure of simplex counts, Betti numbers, and
    normalized Betti numbers over time (see _yearly_xs_and_series):
      Row 1: N0, N1, N2
      Row 2: beta_0, beta_1, beta_2
      Row 3: beta_tilde_0, beta_tilde_1, beta_tilde_2

    Parameters
    ----------
    Betti : ndarray, shape (4, n_steps)
      Betti[k][t] is the k-th Betti number at step t.
    SimplexCounts : list of list of float
      Simplex counts by size at each step.
    Euler : ndarray, shape (n_steps,)
      Euler characteristic at each step (unused by this panel set,
      passed through to _yearly_xs_and_series).
    cycleR : array-like, length n_steps
      CR_1 at each step.
    dates : list of str
      Date string (year in the first 4 characters) for each step.
    title : str
      Figure title.
    save_path : str or None
      If given, filename prefix (without extension) the figure is
      saved to as PDF and PNG.

    Returns
    -------
    fig : matplotlib.figure.Figure
      The created figure.
    """
    xs, Y = _yearly_xs_and_series(Betti, SimplexCounts, Euler, cycleR, dates)
    xlabel = 'Year'

    panels = [
        (Y['s0'],  r'$N_0$ (nodes)'),
        (Y['s1'],  r'$N_1$ (edges)'),
        (Y['s2'],  r'$N_2$ (triangles)'),
        (Y['b0'],  r'$\beta_0$ (components)'),
        (Y['b1'],  r'$\beta_1$ (tunnels)'),
        (Y['b2'],  r'$\beta_2$ (voids)'),
        (Y['tb0'], r'$\tilde{\beta}_0 = \beta_0 / N_0$'),
        (Y['tb1'], r'$\tilde{\beta}_1 = \beta_1 / N_1$'),
        (Y['tb2'], r'$\tilde{\beta}_2 = \beta_2 / N_2$'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    for ax, (y, label) in zip(axes.flatten(), panels):
        _style_ax(ax, xs, y, xlabel)
        ax.set_title(label, pad=6)

    if save_path:
        fig.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# ── Figure 2: CR1-b1 g1 | CR2-b2 g2 | Euler (centred) ───────────────────────

def plot_ph_cyclerank_euler(Betti, SimplexCounts, Euler, cycleR, dates,
                            title='', save_path=None):
    """
    Plots cycle rank, filling efficiency, and Euler characteristic
    over time (see _yearly_xs_and_series). gamma_k is the fraction of
    ADDED (k+1)-simplices that filled a k-cycle, not the fraction of
    k-cycles filled. Layout:
      Row 1 (2 cols): CR_1 - beta_1,  gamma_1 = (CR_1 - beta_1) / N_2
      Row 2 (2 cols): CR_2 - beta_2,  gamma_2 = (CR_2 - beta_2) / N_3
      Row 3 (1 col, centred): Euler characteristic

    Parameters
    ----------
    Betti : ndarray, shape (4, n_steps)
      Betti[k][t] is the k-th Betti number at step t.
    SimplexCounts : list of list of float
      Simplex counts by size at each step.
    Euler : ndarray, shape (n_steps,)
      Euler characteristic at each step.
    cycleR : array-like, length n_steps
      CR_1 at each step.
    dates : list of str
      Date string (year in the first 4 characters) for each step.
    title : str
      Figure title.
    save_path : str or None
      If given, filename prefix (without extension) the figure is
      saved to as PDF and PNG.

    Returns
    -------
    fig : matplotlib.figure.Figure
      The created figure.
    """
    xs, Y = _yearly_xs_and_series(Betti, SimplexCounts, Euler, cycleR, dates)
    xlabel = 'Year'

    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(title, fontsize=14)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)

    # Row 1: 2 panels spanning cols 0-1 and 2-3
    ax_cr1b1 = fig.add_subplot(gs[0, :2])
    ax_g1    = fig.add_subplot(gs[0, 2:])

    # Row 2: 2 panels
    ax_cr2b2 = fig.add_subplot(gs[1, :2])
    ax_g2    = fig.add_subplot(gs[1, 2:])

    # Row 3: 1 centred panel spanning middle 2 cols
    ax_chi   = fig.add_subplot(gs[2, 1:3])

    panels = [
        (ax_cr1b1, Y['cr1b1'], r'$\mathcal{CR}_1 - \beta_1$ (filled tunnels)'),
        (ax_g1,    Y['g1'],    r'$\gamma_1 = (\mathcal{CR}_1 - \beta_1)\,/\,N_2$'),
        (ax_cr2b2, Y['cr2b2'], r'$\mathcal{CR}_2 - \beta_2$ (filled voids)'),
        (ax_g2,    Y['g2'],    r'$\gamma_2 = (\mathcal{CR}_2 - \beta_2)\,/\,N_3$'),
        (ax_chi,   Y['chi'],   r'$\chi$ (Euler characteristic)'),
    ]

    for ax, y, label in panels:
        _style_ax(ax, xs, y, xlabel)
        ax.set_title(label, pad=6)

    if save_path:
        fig.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# ── Figure 3: N0 N1 | b0 b1 | tb0 tb1 ───────────────────────────────────────

def plot_ph_graph(Betti, SimplexCounts, Euler, cycleR, dates,
                  title='', save_path=None):
    """
    Plots a 3x2 figure restricted to graph-level (dim 0 and 1)
    quantities (see _yearly_xs_and_series):
      Row 1: N0, N1
      Row 2: beta_0, beta_1
      Row 3: beta_tilde_0, beta_tilde_1

    Parameters
    ----------
    Betti : ndarray, shape (4, n_steps)
      Betti[k][t] is the k-th Betti number at step t.
    SimplexCounts : list of list of float
      Simplex counts by size at each step.
    Euler : ndarray, shape (n_steps,)
      Euler characteristic at each step (unused by this panel set,
      passed through to _yearly_xs_and_series).
    cycleR : array-like, length n_steps
      CR_1 at each step.
    dates : list of str
      Date string (year in the first 4 characters) for each step.
    title : str
      Figure title.
    save_path : str or None
      If given, filename the figure is saved to as PNG.

    Returns
    -------
    fig : matplotlib.figure.Figure
      The created figure.
    """
    xs, Y  = _yearly_xs_and_series(Betti, SimplexCounts, Euler, cycleR, dates)
    xlabel = 'Year'

    panels = [
        (Y['s0'],  r'$N_0$ (vertices)'),
        (Y['s1'],  r'$N_1$ (edges)'),
        (Y['b0'],  r'$\beta_0$ (components)'),
        (Y['b1'],  r'$\beta_1$ (tunnels)'),
        (Y['tb0'], r'$\tilde{\beta}_0 = \beta_0 / N_0$'),
        (Y['tb1'], r'$\tilde{\beta}_1 = \beta_1 / N_1$'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    for ax, (y, label) in zip(axes.flatten(), panels):
        _style_ax(ax, xs, y, xlabel)
        ax.set_title(label, pad=6)

    if save_path:
        fig.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# ── convenience wrapper matching original plot_ph_yearly signature ────────────

def plot_ph_yearly(Betti, SimplexCounts, Euler, cycleR, dates,
                   title='', save_path='ph_yearly'):
    """
    Convenience wrapper that produces both standard yearly PH figures
    by calling plot_ph_simplices_betti and plot_ph_cyclerank_euler in
    sequence.

    Parameters
    ----------
    Betti : ndarray, shape (4, n_steps)
      Betti[k][t] is the k-th Betti number at step t.
    SimplexCounts : list of list of float
      Simplex counts by size at each step.
    Euler : ndarray, shape (n_steps,)
      Euler characteristic at each step.
    cycleR : array-like, length n_steps
      CR_1 at each step.
    dates : list of str
      Date string (year in the first 4 characters) for each step.
    title : str
      Title used for both figures.
    save_path : str
      Filename prefix; '_fig1'/'_fig2' are appended for the two figures.

    Returns
    -------
    None
      Displays and saves both figures.
    """
    plot_ph_simplices_betti(Betti, SimplexCounts, Euler, cycleR, dates,
                            title=title,
                            save_path=save_path + '_fig1')
    plot_ph_cyclerank_euler(Betti, SimplexCounts, Euler, cycleR, dates,
                            title=title,
                            save_path=save_path + '_fig2')

def ZZ_LittleSIS(rel_dicts, timing=False, maxDim=np.inf, random_seed=42):
    """
    Builds a zigzag filtration from LittleSIS relationship dicts. Edges
    are added at start_date and removed at end_date (if present and
    is_current is False). Duplicate edges (same undirected pair) are
    merged if their intervals overlap, and kept as separate events if
    they do not. Isolated nodes are removed when their last edge is
    deleted. Tracks clique-complex simplex counts and cycle rank CR_1
    (= N_1 - N_0 + beta_0) at each event, then computes zigzag
    persistent homology from the resulting filtration.

    Parameters
    ----------
    rel_dicts : list of dict
        Relationship dictionaries, e.g. [has_start_p2p, has_start_end_p2p].
    timing : bool
        If True, print progress messages and timing at each stage.
    maxDim : int or np.inf
        Maximum simplex dimension to track (e.g. maxDim=2 tracks up to triangles).
    random_seed : int
        Seed for tie-breaking among events sharing the same date.

    Returns
    -------
    Betti        : np.ndarray, shape (4, n_steps)
    SimplexCounts: list of lists, one entry per event step
    Euler        : np.ndarray, shape (n_steps,)
    cycleR       : list, length n_steps; CR_1 at each step
    dates        : list of str, the event date string for each step
    """
    rng = random.Random(random_seed)
    SENTINEL = '9999-99-99'  # Represents open-ended intervals

    # -------------------------------------------------------------------------
    # (1/5) Collect intervals, merge overlapping ones, build event list
    # -------------------------------------------------------------------------
    if timing:
        print("(1/5) Collecting and sorting events", flush=True)
        t0 = time.time()

    # Group all intervals by undirected pair
    pair_intervals = {}
    for rel_dict in rel_dicts:
        for rel in rel_dict.values():
            start = rel.get('start_date')
            if not start:
                continue
            u, v = rel['entity1_id'], rel['entity2_id']
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            # None end means open-ended (is_current=True or no end_date)
            end = (rel.get('end_date')
                   if not rel.get('is_current', False) else None)
            if key not in pair_intervals:
                pair_intervals[key] = []
            pair_intervals[key].append((start, end))

    def merge_intervals(intervals):
        """
        Standard interval merge. Intervals with None end extend to SENTINEL.
        Returns list of (start, end) with None restored for open-ended intervals.
        Non-overlapping intervals are kept as separate entries.
        """
        ivs = sorted(
            [(s, e if e is not None else SENTINEL) for s, e in intervals],
            key=lambda x: x[0]
        )
        merged = []
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:
                # Intervals overlap: merge into earlier start, later end
                cur_e = max(cur_e, e)
            else:
                # No overlap: keep as separate interval
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return [(s, None if e == SENTINEL else e) for s, e in merged]

    # Build event list: (date_str, priority, rand_key, action, u, v)
    # priority 0=add, 1=remove so additions precede removals on same date
    raw_events = []
    for (u, v), intervals in pair_intervals.items():
        for start, end in merge_intervals(intervals):
            raw_events.append((start, 0, rng.random(), 'add', u, v))
            if end is not None:
                raw_events.append((end, 1, rng.random(), 'remove', u, v))

    raw_events.sort(key=lambda x: (x[0], x[1], x[2]))
    n_steps = len(raw_events)
    dates   = [e[0] for e in raw_events]

    if timing:
        n_add = sum(1 for e in raw_events if e[3] == 'add')
        n_rem = n_steps - n_add
        print(f"  {n_add} additions, {n_rem} removals ({n_steps} total). "
              f"Time: {time.time()-t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # (2/5) Initialize graph and data structures
    # -------------------------------------------------------------------------
    if timing:
        print("(2/5) Initializing graph and data structures", flush=True)
        t0 = time.time()

    G             = nx.Graph()
    Times         = {}
    SimplexCounts = []
    cycleR        = np.zeros(n_steps)

    if timing:
        print(f"  Done. Time: {time.time()-t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # (3/5) Evolve graph event by event
    # -------------------------------------------------------------------------
    if timing:
        print("(3/5) Evolving graph", flush=True)
        t0 = time.time()

    iterator = enumerate(tqdm(raw_events)) if timing else enumerate(raw_events)

    for timer, (date_str, _, __, action, u, v) in iterator:

        current_counts = SimplexCounts[-1].copy() if SimplexCounts else []

        # --- ADDITION --------------------------------------------------------
        if action == 'add':
            new_simplices = set()

            for node in [u, v]:
                if node not in G:
                    G.add_node(node)
                    new_simplices.add((node,))

            common = list(nx.common_neighbors(G, u, v))
            G.add_edge(u, v)
            new_simplices.add(tuple(sorted([u, v])))

            if common and maxDim > 1:
                H = G.subgraph([u, v] + common)
                for clique in nx.find_cliques(H):
                    if u in clique and v in clique:
                        clique_s = sorted(clique)
                        for size in range(3, min(len(clique_s),
                                                  maxDim) + 1):
                            for face in combinations(clique_s, size):
                                if u in face and v in face:
                                    new_simplices.add(face)

            for simplex in new_simplices:
                dim = len(simplex) - 1
                if dim > maxDim:
                    continue
                while len(current_counts) <= dim:
                    current_counts.append(0)
                current_counts[dim] += 1
                if dim <= 3:
                    if simplex in Times:
                        Times[simplex].append(timer)
                    else:
                        Times[simplex] = [timer]

        # --- REMOVAL ---------------------------------------------------------
        else:
            if not G.has_edge(u, v):
                SimplexCounts.append(current_counts)
                cycleR[timer] = (G.number_of_edges() - G.number_of_nodes()
                                 if G.number_of_nodes() > 0 else 0)
                continue

            # Collect all simplices containing (u,v) before removing edge
            rem_simplices = set([tuple(sorted([u, v]))])
            for clique in nx.find_cliques(G, sorted([u, v])):
                remainder = [node for node in clique
                             if node != u and node != v]
                for r in range(1, min(len(remainder),
                                      maxDim - 1) + 1):
                    for face in combinations(remainder, r):
                        rem_simplices.add(tuple(sorted(face + (u, v))))

            G.remove_edge(u, v)

            # Remove isolated nodes
            for node in [u, v]:
                if node in G and G.degree(node) == 0:
                    G.remove_node(node)
                    iso = (node,)
                    rem_simplices.add(iso)

            for simplex in rem_simplices:
                dim = len(simplex) - 1
                if dim > maxDim:
                    continue
                if dim < len(current_counts):
                    current_counts[dim] -= 1
                if dim <= 3 and simplex in Times:
                    Times[simplex].append(timer)

        SimplexCounts.append(current_counts)
        cycleR[timer] = (G.number_of_edges() - G.number_of_nodes()
                         if G.number_of_nodes() > 0 else 0)

    if timing:
        print(f"  Evolution complete. Time: {time.time()-t0:.2f}s", flush=True)

    # Close all simplices still open at the final step
    for tlist in Times.values():
        if len(tlist) % 2 == 1:
            tlist.append(n_steps)

    # -------------------------------------------------------------------------
    # (4/5) Zigzag Persistent Homology
    # -------------------------------------------------------------------------
    if timing:
        print("(4/5) Computing Zigzag Persistent Homology", flush=True)
        t0 = time.time()

    simplices = [list(k) for k in Times]
    times     = [Times[k] for k in Times]
    del Times; gc.collect()

    f = d.Filtration(simplices)
    zz, dgms, cells = d.zigzag_homology_persistence(f, times)
    del simplices; del times; gc.collect()

    if timing:
        print(f"  ZZ PH complete. Time: {time.time()-t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # (5/5) Betti numbers and Euler characteristic
    # -------------------------------------------------------------------------
    if timing:
        print("(5/5) Extracting Betti numbers and Euler characteristic",
              flush=True)
        t0 = time.time()

    Betti = np.zeros((4, n_steps))
    for i, dgm in enumerate(dgms):
        if i >= 4:
            break
        for p in dgm:
            birth = int(p.birth)
            death = int(min(p.death, n_steps))
            Betti[i][birth:death] += 1

    Euler = np.zeros(n_steps)
    for i, counts in enumerate(SimplexCounts):
        for j, c in enumerate(counts):
            Euler[i] += (-1)**j * c

    cycleR = [cycleR[i] + Betti[0][i] for i in range(n_steps)]

    if timing:
        print(f"  Done. Time: {time.time()-t0:.2f}s", flush=True)

    return Betti, SimplexCounts, Euler, cycleR, dates

def _yearly(arr, dates):
    """
    Reduces a per-step time series to one end-of-year snapshot per
    year spanned by dates, taking each year's last observed value.
    Unlike _aggregate_yearly, also derives the year range from dates.

    Parameters
    ----------
    arr : ndarray
      Per-step values.
    dates : list of str
      Date string (year in the first 4 characters) for each step,
      aligned with arr by index.

    Returns
    -------
    yr_range : ndarray
      Calendar years spanned by dates, one entry per year.
    out : ndarray
      One value per entry in yr_range: the last arr value in that
      year, or NaN if the year has no steps.
    """
    years    = np.array([int(d[:4]) for d in dates])
    yr_range = np.arange(years.min(), years.max() + 1)
    out      = np.full(len(yr_range), np.nan)
    for i, yr in enumerate(yr_range):
        mask = years == yr
        if np.any(mask):
            out[i] = arr[mask][-1]
    return yr_range, out

def _prepare(Betti, SC, Euler, cycleR, dates):
    """
    Computes graph-level normalized Betti numbers and aggregates
    simplex counts, Betti numbers, normalized Betti numbers, and
    Euler characteristic to end-of-year snapshots, for use in cohort
    comparison plots.

    Parameters
    ----------
    Betti : ndarray, shape (4, n_steps)
      Betti[k][t] is the k-th Betti number at step t.
    SC : list of list of float
      Simplex counts by size at each step.
    Euler : ndarray, shape (n_steps,)
      Euler characteristic at each step.
    cycleR : array-like, length n_steps
      CR_1 at each step (unused by this function's output, accepted
      for signature symmetry with other _yearly_xs_and_series callers).
    dates : list of str
      Date string (year in the first 4 characters) for each step.

    Returns
    -------
    dict
      Maps each of 's0','s1','b0','b1','tb0','tb1','chi' to a
      (years, values) tuple from _yearly, where tb0/tb1 are the
      normalized Betti numbers beta_tilde_0 = beta_0/N_0 and
      beta_tilde_1 = beta_1/N_1.
    """
    n   = len(SC)
    b0  = Betti[0].astype(float)
    b1  = Betti[1].astype(float)
    s0  = _sc(SC, 0, n)
    s1  = _sc(SC, 1, n)
    chi = np.array(Euler, dtype=float)

    with np.errstate(invalid='ignore', divide='ignore'):
        tb0 = np.where(s0 > 0, b0 / s0, np.nan)
        tb1 = np.where(s1 > 0, b1 / s1, np.nan)

    raw = dict(s0=s0, s1=s1, b0=b0, b1=b1, tb0=tb0, tb1=tb1, chi=chi)
    return {k: _yearly(v, dates) for k, v in raw.items()}

def line_kw(color):
    """
    Builds this module's standard matplotlib line/marker style kwargs
    for a given color.

    Parameters
    ----------
    color : str
      Line and marker edge color.

    Returns
    -------
    dict
      Keyword arguments suitable for Axes.plot (color, lw, marker,
      markersize, markerfacecolor, markeredgecolor, markeredgewidth).
    """
    return dict(color=color, lw=1.4, marker='o', markersize=3.5,
                markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=1.0)

def _style_ax2(ax, B, N, C, label, labels):
    """
    Plots three cohort series (bulk, non-bulk, combined) as
    year-indexed lines on one axis, with this module's standard style.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
      Axis to draw on.
    B : tuple
      (years, values) for the "bulk" cohort series (see _yearly).
    N : tuple
      (years, values) for the "non-bulk" cohort series.
    C : tuple
      (years, values) for the "combined" cohort series.
    label : str
      Subplot title.
    labels : sequence
      labels[0:3] are the legend labels for B, N, C respectively;
      labels[3:6] are their corresponding line colors.

    Returns
    -------
    None
      Draws onto ax and sets its title/axis formatting.
    """
    yrs_b, vals_b = B
    yrs_n, vals_n = N
    yrs_c, vals_c = C
    ax.plot(yrs_b, vals_b, label=labels[0],     **line_kw(labels[3]))
    ax.plot(yrs_n, vals_n, label=labels[1],  **line_kw(labels[4]))
    ax.plot(yrs_c, vals_c, label=labels[2], **line_kw(labels[5]))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v:,.3g}'))
    ax.set_title(label, pad=6)

def plot_cohort_graph(labels, Betti_bulk, SC_bulk, Euler_bulk, cycleR_bulk, dates_bulk,
                      Betti_non,  SC_non,  Euler_non,  cycleR_non,  dates_non,
                      Betti_comb, SC_comb, Euler_comb, cycleR_comb, dates_comb,
                      title='', save_path=None):
    """
    Plots simplex counts, Betti numbers, normalized Betti numbers, and
    Euler characteristic over time for three cohorts (e.g. bulk-import
    vs. non-bulk vs. combined relationships) overlaid on shared axes
    (see _prepare and _style_ax2). Layout:
      Row 1 (2 cols): N0,      N1
      Row 2 (2 cols): beta_0,  beta_1
      Row 3 (2 cols): beta_tilde_0, beta_tilde_1
      Row 4 (1 col, centred): Euler characteristic
    Single shared legend beneath the suptitle.

    Parameters
    ----------
    labels : sequence
      labels[0:3] are legend labels for the bulk/non-bulk/combined
      cohorts; labels[3:6] are their corresponding line colors (see
      _style_ax2).
    Betti_bulk, SC_bulk, Euler_bulk, cycleR_bulk, dates_bulk :
      Betti array, simplex counts, Euler characteristic, CR_1, and
      dates (see PH_LittleSIS/ZZ_LittleSIS) for the "bulk" cohort.
    Betti_non, SC_non, Euler_non, cycleR_non, dates_non :
      Same, for the "non-bulk" cohort.
    Betti_comb, SC_comb, Euler_comb, cycleR_comb, dates_comb :
      Same, for the "combined" cohort.
    title : str
      Figure title.
    save_path : str or None
      If given, filename prefix (without extension) the figure is
      saved to as PDF and PNG.

    Returns
    -------
    fig : matplotlib.figure.Figure
      The created figure.
    """
    B = _prepare(Betti_bulk, SC_bulk, Euler_bulk, cycleR_bulk, dates_bulk)
    N = _prepare(Betti_non,  SC_non,  Euler_non,  cycleR_non,  dates_non)
    C = _prepare(Betti_comb, SC_comb, Euler_comb, cycleR_comb, dates_comb)

    fig = plt.figure(figsize=(11, 13))
    fig.suptitle(title, fontsize=14, y=0.98)
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.35,
                            left=0.08, right=0.97, top=0.91, bottom=0.06)

    ax_s0  = fig.add_subplot(gs[0, :2])
    ax_s1  = fig.add_subplot(gs[0, 2:])
    ax_b0  = fig.add_subplot(gs[1, :2])
    ax_b1  = fig.add_subplot(gs[1, 2:])
    ax_tb0 = fig.add_subplot(gs[2, :2])
    ax_tb1 = fig.add_subplot(gs[2, 2:])
    ax_chi = fig.add_subplot(gs[3, 1:3])

    panels = [
        (ax_s0,  's0',  r'$N_0$ (nodes)'),
        (ax_s1,  's1',  r'$N_1$ (edges)'),
        (ax_b0,  'b0',  r'$\beta_0$ (components)'),
        (ax_b1,  'b1',  r'$\beta_1$ (tunnels)'),
        (ax_tb0, 'tb0', r'$\tilde{\beta}_0 = \beta_0 / N_0$'),
        (ax_tb1, 'tb1', r'$\tilde{\beta}_1 = \beta_1 / N_1$'),
        (ax_chi, 'chi', r'$\chi$ (Euler characteristic)'),
    ]

    for ax, key, label in panels:
        _style_ax2(ax, B[key], N[key], C[key], label, labels)

    # Single shared legend placed between suptitle and top row of panels
    handles, labels = ax_s0.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.0),
               fontsize=10, framealpha=0.9, edgecolor='#cccccc')

    if save_path:
        fig.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig