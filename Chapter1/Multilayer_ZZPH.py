"""
Module Name: Multilayer_ZZPH.py
Description: Contains functions for applying multilayer
zigzag persistence to a toy example

Author: Jason LaRuez
Date: 2026
"""

# ============================================================================
# IMPORTS
# ============================================================================
import networkx as nx # Network structures
import numpy as np # Numpy arrays and operations
import random # Random sampling for network models
from itertools import combinations, product # For getting different simplices and all combinations of lists

import matplotlib.pyplot as plt # Plotting
import time # Timing simulations
from tqdm.notebook import trange, tqdm # Allows for real-time progress bar of simulations

import gc # Memory management
import pickle # Takes environment variables and saves them as is
import gzip # Allows for compression of saved files
from joblib import Parallel, delayed # Parallelization functions
import multiprocessing # Get number of cpu cores

import os # Iterating over directories for files
import math 
import seaborn as sns # Heatmap plots

import dionysus as d # C++ package with python bindings for persistent homology

# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================
# ZZPH Code
# ============================================================================

def Windowing(width, layer, Edges):
    """
    Given a sequence of edge sets, form windowings of these edge
    sets where "width" is the initial width used in multilayer
    ZZPH, and layer (0, 1, ...) is the layer in the lattice for which
    the windowing is being constructed.
    """
    # The i-th window in any layer always contains the i-th window from the 0-layer
    W = [edges.copy() for i, edges in enumerate(Edges) if i < int(len(Edges)/width) - layer]

    # Take unions of neighboring windows based on the width and current layer
    for idx in range(len(W)):
        for j in range(1, layer+1):
            W[idx].update(Edges[idx+j])

    return W

def remap(window_idx, width, layer, numWindows):
    return ( window_idx + (width / 2) * (layer + 1) ) / numWindows

def ZZPH(Edges, width = None, layer = None):
    """
    Given a sequence of edge sets, form a sequence of graphs,
    then for each graph construct the corresponding clique complex.
    Then, ZZPH is computed on this sequence of complexes and the 
    resultant persistence diagrams are returned. If the width and
    layer from MLZZPH is provided, then the scale of the diagrams are
    remapped to [0,1].
    """
    # Initialize T for times simplices are added and removed for ZZPH
    Times = dict()
    # Keep track of simplices from previous timestep to see track if
    # simplices remain or are removed in neighboring timesteps
    lastSimplices = set()

    for i, step_edges in enumerate(Edges):
        Simplices = set()

        # For graph from edge set and find maximal cliques
        G = nx.Graph()
        G.add_edges_from(step_edges)
        cliques = list(nx.find_cliques(G))

        # Iterate over maximal cliques to enumerate all simplices (dim 3 or less) which
        # are present in the current windowed graph
        for clique in cliques:
            numNodes = len(clique)
            clique = sorted(clique)
            for r in range(1, min(numNodes, 4)+1):
                # combinations(S,r) is from itertools and returns iterator corresponding
                # to all r-subsets of S.
                for face in combinations(clique, r):
                    if face not in Simplices:
                        Simplices.add(face)
                        # In this case the face had previously been added and removed, so we add it again
                        if (face in Times) and (face not in lastSimplices):
                            Times[face].append(i)
                        # In this case the face had never existed, so we add it for the first time
                        elif face not in Times:
                            Times[face] = [i]

        # Checking if simplices from prior timestep persisted
        for simplex in lastSimplices:
            # The simplex existed in prior timestep but not in current, so
            # it exists in T. We append removal time to T. Otherwise, the simplex
            # is still present in the current timestep so we do not update T yet
            if simplex not in Simplices:
                Times[simplex].append(i)

        # Update prior simplices to current timestep
        lastSimplices = Simplices.copy()

    # Remove all simplices at the end to avoid infinitely persistent features
    for simplex in lastSimplices:
        Times[simplex].append(len(Edges))

    # Extract list of every simplex added/removed, and list of times they were
    # added/removed, for input into zigzag persistence.
    simplices = [list(key) for key in Times]; 
    if width:
        times = [[remap(Times[key][i],width,layer,len(Edges)+layer) for i in range(len(Times[key]))] for key in Times]
    else:
        times = [Times[key] for key in Times]

    # Construct filtration and compute homology
    f = d.Filtration(simplices)
    zz, dgms, cells = d.zigzag_homology_persistence(f, times)

    return dgms


# ============================================================================
# Toy Example Code
# ============================================================================

def PeriodicRing(N, p_ER, max_p_WS, k_WS, period, steps):
    """
    N = number of nodes
    p_ER = constant probability of a non-ring edge being added
    max_p_WS = maximum of the sinusoidal probability of including
    an edge from the Watts-strogatz ring lattice
    period = number of time steps for 1 sinusoidal period of the ring
    lattice probabilities
    steps = number of time steps to simulate
    """
    # Initialize lists of lattice edges and non-lattice edges
    WS_edges = set(tuple(sorted([i,(i+j)%N])) for i in range(N) for j in range(1,int(k_WS / 2)+1))
    ER_edges = set(edge for edge in combinations(range(N),2) if edge not in WS_edges)
    # Keep track of edge set at each time step
    Edges = [set() for _ in range(steps)]

    for step in range(steps):
        # For each non-lattice edge, add to edges with prob p_ER
        for edge in ER_edges:
            if random.random() < p_ER:
                Edges[step].add(edge)

        # For each lattice edge, add to edges with sinusoidal probability bounded
        # between 0 and max_p_WS with period = period
        p_WS = 0.5 * (max_p_WS) * math.sin(2*np.pi * step / period) + 0.5 * (max_p_WS)
        for edge in WS_edges:
            if random.random() < p_WS:
                Edges[step].add(edge)

    return Edges

def compute_distance_matrices(layers, period, DGMs, p):
    """ * Written using AI *
    Computes D0 and D1 Wasserstein or Bottleneck distance matrices
    between persistence diagrams for given layers and period.

    Parameters:
    - layers: A range or list of layer indices.
    - period: The current period being processed (used to access DGMs).
    - DGMs: Dictionary containing persistence diagrams (DGMs[period][layer][dim]).
    - p: String indicating the distance type ('1' for 1-Wasserstein, '2' for 2-Wasserstein, 'inf' for Bottleneck).

    Returns:
    - d0: The computed distance matrix for dimension 0.
    - d1: The computed distance matrix for dimension 1.
    """
    n_layers = len(layers)
    d0 = np.zeros((n_layers, n_layers))
    d1 = np.zeros((n_layers, n_layers))

    def calculate_distances_for_chunk(chunk, period_data, p_type):
        """Process a list of (i,j) pairs, returning a list of (i,j,dist0,dist1)."""
        results = []
        for i, j in chunk:
            dgm_i_0 = period_data[i][0]
            dgm_j_0 = period_data[j][0]
            dgm_i_1 = period_data[i][1] if len(period_data[i]) > 1 else d.Diagram()
            dgm_j_1 = period_data[j][1] if len(period_data[j]) > 1 else d.Diagram()

            dist0, dist1 = 0.0, 0.0
            if p_type == '1':
                dist0 = d.wasserstein_distance(dgm_i_0, dgm_j_0, q=1, internal_p=-1)
                dist1 = d.wasserstein_distance(dgm_i_1, dgm_j_1, q=1, internal_p=-1)
            elif p_type == '2':
                dist0 = d.wasserstein_distance(dgm_i_0, dgm_j_0, q=2, internal_p=-2)
                dist1 = d.wasserstein_distance(dgm_i_1, dgm_j_1, q=2, internal_p=-2)
            elif p_type == 'inf':
                dist0 = d.bottleneck_distance(dgm_i_0, dgm_j_0)
                dist1 = d.bottleneck_distance(dgm_i_1, dgm_j_1)
            else:
                raise ValueError("Invalid value for 'p'. Must be '1', '2', or 'inf'.")
            results.append((i, j, dist0, dist1))
        return results

    # Generate all unique (i, j) pairs for upper triangle
    pairs = [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)]

    # Split pairs into one chunk per worker
    n_jobs = multiprocessing.cpu_count()
    chunks = [pairs[k::n_jobs] for k in range(n_jobs)]

    # Each worker receives a whole chunk to loop through internally
    results_nested = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(calculate_distances_for_chunk)(chunk, DGMs[period], p)
        for chunk in tqdm(chunks, desc=f"Calculating {p}-distance for Period {period}")
    )

    # Flatten results and populate distance matrices
    for chunk_results in results_nested:
        for i, j, dist0, dist1 in chunk_results:
            d0[i, j] = dist0; d0[j, i] = dist0
            d1[i, j] = dist1; d1[j, i] = dist1

    return d0, d1

def plot_period_group_publication(period_group, D0, D1, p_max_WS, figsize=(10, 13), name = None):
    nrows = len(period_group)

    # Collect matrices so each homology column uses a fixed shared color scale
    all_d0 = [np.asarray(D0[p]['1']) for p in period_group]
    all_d1 = [np.asarray(D1[p]['1']) for p in period_group]

    vmin0 = min(arr.min() for arr in all_d0)
    vmax0 = max(arr.max() for arr in all_d0)
    vmin1 = min(arr.min() for arr in all_d1)
    vmax1 = max(arr.max() for arr in all_d1)

    # Figure + axes
    fig, ax = plt.subplots(nrows, 2, figsize=figsize,
        constrained_layout=False, gridspec_kw={"wspace": 0.18, "hspace": 0.18})

    if nrows == 1:
        ax = np.array([ax])

    # Supertitle
    fig.suptitle(f"Ring Model: $p_{{\\max}}$ = {p_max_WS}", fontsize=16, fontweight="bold", y=0.93)

    for i, period in enumerate(period_group):
        d0 = np.asarray(D0[period]['1']); d1 = np.asarray(D1[period]['1'])
        n0 = d0.shape[0]; n1 = d1.shape[0]

        # Choose readable tick spacing automatically
        step0 = max(1, int(np.ceil(n0 / 10))); step1 = max(1, int(np.ceil(n1 / 10)))

        ticks0 = np.arange(0, n0, step0); ticks1 = np.arange(0, n1, step1)

        # ----- Left column: 0-dimensional homology -----
        hm0 = sns.heatmap(
            d0, ax=ax[i, 0], cmap="plasma", vmin=vmin0, vmax=vmax0, 
            cbar=True, cbar_kws={"shrink": 1.0})
        ax[i, 0].invert_yaxis()

        if i == 0:
            ax[i, 0].set_title("0-Dimensional Homology", fontsize=14, pad=6)

        if i == nrows - 1:
            ax[i, 0].set_xlabel("Lattice Layer", fontsize=12)
        else:
            ax[i, 0].set_xlabel("")

        ax[i, 0].set_ylabel("Lattice Layer", fontsize=12)

        ax[i, 0].set_xticks(ticks0 + 0.5)
        ax[i, 0].set_xticklabels(ticks0, rotation=90, fontsize=9)
        ax[i, 0].set_yticks(ticks0 + 0.5)
        ax[i, 0].set_yticklabels(ticks0, rotation=0, fontsize=9)

        # Remove label from left colorbar
        cbar0 = hm0.collections[0].colorbar
        cbar0.ax.set_ylabel("")
        cbar0.ax.tick_params(labelsize=10)

        # Horizontal period label with gray box behind it
        ax[i, 0].text(-0.25, 0.5, f"Period = {period}", transform=ax[i, 0].transAxes, rotation=90,
            va="center",  ha="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="gray", alpha=1.0))

        # ----- Right column: 1-dimensional homology -----
        hm1 = sns.heatmap(d1, ax=ax[i, 1], cmap="plasma", vmin=vmin1, vmax=vmax1,
            cbar=True, cbar_kws={"label": "1-Wasserstein Distance", "shrink": 1.0})
        ax[i, 1].invert_yaxis()

        if i == 0:
            ax[i, 1].set_title("1-Dimensional Homology", fontsize=14, pad=6)

        if i == nrows - 1:
            ax[i, 1].set_xlabel("Lattice Layer", fontsize=12)
        else:
            ax[i, 1].set_xlabel("")

        ax[i, 1].set_ylabel("")

        ax[i, 1].set_xticks(ticks1 + 0.5)
        ax[i, 1].set_xticklabels(ticks1, rotation=90, fontsize=9)
        ax[i, 1].set_yticks(ticks1 + 0.5)
        ax[i, 1].set_yticklabels(ticks1, rotation=0, fontsize=9)

        cbar1 = hm1.collections[0].colorbar
        cbar1.ax.set_ylabel("1-Wasserstein Distance", fontsize=12, rotation=270, labelpad=16)
        cbar1.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0.10, 0.04, 0.98, 0.91])
    if name:
        plt.savefig(name)
    plt.show()

def plot_persistence_diagram_publication(diagram, ax=None, title=None, point_size=42,
    point_color="#123B6D", diag_color="0.55", tick_fontsize=11, label_fontsize=13, title_fontsize=14):
    """
    Plot a Dionysus persistence diagram in a publication-style format.
    Parameters
    ----------
    diagram : dionysus diagram
        A persistence diagram, e.g. DGMs[period][layer][dim]
    ax : matplotlib.axes.Axes or None
        Axis to draw on. If None, creates a new figure.
    title : str or None
        Optional subplot title.
    point_size : float
        Scatter marker size.
    point_color : str
        Color of persistence points.
    diag_color : str
        Color of diagonal y=x line.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

    births = []
    deaths = []

    # Extract finite points only
    for pt in diagram:
        b = float(pt.birth)
        de = float(pt.death)

        if np.isfinite(b) and np.isfinite(de):
            births.append(b)
            deaths.append(de)

    births = np.array(births)
    deaths = np.array(deaths)

    # Scatter points
    if len(births) > 0:
        ax.scatter(births, deaths, s=point_size, color=point_color, 
                   edgecolors="black", linewidths=0.35, zorder=3)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle="-", linewidth=1.4, color=diag_color, zorder=1)

    # Fixed axes requested
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Labels
    ax.set_xlabel("Normalized Birth Time", fontsize=label_fontsize)
    ax.set_ylabel("Normalized Death Time", fontsize=label_fontsize)

    # Optional title
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, pad=8)

    # Clean styling
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    return ax

def plot_period_group_publication_pmax(p_group, D0, D1, period, figsize=(10, 13), name = None):
    nrows = len(p_group)

    # Collect matrices so each homology column uses a fixed shared color scale
    all_d0 = [np.asarray(D0[p]['1']) for p in p_group]
    all_d1 = [np.asarray(D1[p]['1']) for p in p_group]

    vmin0 = min(arr.min() for arr in all_d0)
    vmax0 = max(arr.max() for arr in all_d0)
    vmin1 = min(arr.min() for arr in all_d1)
    vmax1 = max(arr.max() for arr in all_d1)

    # Figure + axes
    fig, ax = plt.subplots(
        nrows, 2, figsize=figsize, constrained_layout=False, gridspec_kw={"wspace": 0.18, "hspace": 0.18}
    )

    if nrows == 1:
        ax = np.array([ax])

    # Supertitle
    fig.suptitle(
        f"Ring Model: Period = {period}", fontsize=16, fontweight="bold", y=0.93
    )

    for i, p in enumerate(p_group):
        d0 = np.asarray(D0[p]['1'])
        d1 = np.asarray(D1[p]['1'])

        n0 = d0.shape[0]
        n1 = d1.shape[0]

        # Choose readable tick spacing automatically
        step0 = max(1, int(np.ceil(n0 / 10)))
        step1 = max(1, int(np.ceil(n1 / 10)))

        ticks0 = np.arange(0, n0, step0)
        ticks1 = np.arange(0, n1, step1)

        # ----- Left column: 0-dimensional homology -----
        hm0 = sns.heatmap(
            d0, ax=ax[i, 0],  cmap="plasma", vmin=vmin0,
            vmax=vmax0, cbar=True, cbar_kws={"shrink": 1.0}
        )
        ax[i, 0].invert_yaxis()

        if i == 0:
            ax[i, 0].set_title("0-Dimensional Homology", fontsize=14, pad=6)

        if i == nrows - 1:
            ax[i, 0].set_xlabel("Lattice Layer", fontsize=12)
        else:
            ax[i, 0].set_xlabel("")

        ax[i, 0].set_ylabel("Lattice Layer", fontsize=12)

        ax[i, 0].set_xticks(ticks0 + 0.5)
        ax[i, 0].set_xticklabels(ticks0, rotation=90, fontsize=9)
        ax[i, 0].set_yticks(ticks0 + 0.5)
        ax[i, 0].set_yticklabels(ticks0, rotation=0, fontsize=9)

        # Remove label from left colorbar
        cbar0 = hm0.collections[0].colorbar
        cbar0.ax.set_ylabel("")
        cbar0.ax.tick_params(labelsize=10)

        # Horizontal period label with gray box behind it
        ax[i, 0].text(
            -0.25, 0.5, f"$p_{{\\max}}$ = {p}", transform=ax[i, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="gray", alpha=1.0
            )
        )

        # ----- Right column: 1-dimensional homology -----
        hm1 = sns.heatmap(
            d1, ax=ax[i, 1], cmap="plasma", vmin=vmin1, vmax=vmax1,
            cbar=True, cbar_kws={"label": "1-Wasserstein Distance", "shrink": 1.0}
        )
        ax[i, 1].invert_yaxis()

        if i == 0:
            ax[i, 1].set_title("1-Dimensional Homology", fontsize=14, pad=6)

        if i == nrows - 1:
            ax[i, 1].set_xlabel("Lattice Layer", fontsize=12)
        else:
            ax[i, 1].set_xlabel("")

        ax[i, 1].set_ylabel("")

        ax[i, 1].set_xticks(ticks1 + 0.5)
        ax[i, 1].set_xticklabels(ticks1, rotation=90, fontsize=9)
        ax[i, 1].set_yticks(ticks1 + 0.5)
        ax[i, 1].set_yticklabels(ticks1, rotation=0, fontsize=9)

        cbar1 = hm1.collections[0].colorbar
        cbar1.ax.set_ylabel("1-Wasserstein Distance", fontsize=12, rotation=270, labelpad=16)
        cbar1.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0.10, 0.04, 0.98, 0.91])
    if name:
        plt.savefig(name)
    plt.show()

def plot_period_group_publication_prandom(p_group, D0, D1, period, figsize=(10, 13), name = None):
    nrows = len(p_group)

    # Collect matrices so each homology column uses a fixed shared color scale
    all_d0 = [np.asarray(D0[p]['1']) for p in p_group]
    all_d1 = [np.asarray(D1[p]['1']) for p in p_group]

    # Figure + axes
    fig, ax = plt.subplots(
        nrows, 2, figsize=figsize, constrained_layout=False,
        gridspec_kw={"wspace": 0.18, "hspace": 0.18}
    )

    if nrows == 1:
        ax = np.array([ax])

    # Supertitle
    fig.suptitle(
        f"Ring Model: Period = {period}, $p_{{\\max}}$ = {0.50}",
        fontsize=16, fontweight="bold", y=0.96
    )

    for i, p in enumerate(p_group):
        d0 = np.asarray(D0[p]['1'])
        d1 = np.asarray(D1[p]['1'])

        n0 = d0.shape[0]
        n1 = d1.shape[0]

        # Choose readable tick spacing automatically
        step0 = max(1, int(np.ceil(n0 / 10)))
        step1 = max(1, int(np.ceil(n1 / 10)))

        ticks0 = np.arange(0, n0, step0)
        ticks1 = np.arange(0, n1, step1)

        # ----- Left column: 0-dimensional homology -----
        hm0 = sns.heatmap(
            d0, ax=ax[i, 0], cmap="plasma", cbar=True, cbar_kws={"shrink": 1.0}
        )
        ax[i, 0].invert_yaxis()

        if i == 0:
            ax[i, 0].set_title("0-Dimensional Homology", fontsize=14, pad=6)

        if i == nrows - 1:
            ax[i, 0].set_xlabel("Lattice Layer", fontsize=12)
        else:
            ax[i, 0].set_xlabel("")

        ax[i, 0].set_ylabel("Lattice Layer", fontsize=12)

        ax[i, 0].set_xticks(ticks0 + 0.5)
        ax[i, 0].set_xticklabels(ticks0, rotation=90, fontsize=9)
        ax[i, 0].set_yticks(ticks0 + 0.5)
        ax[i, 0].set_yticklabels(ticks0, rotation=0, fontsize=9)

        # Remove label from left colorbar
        cbar0 = hm0.collections[0].colorbar
        cbar0.ax.set_ylabel("")
        cbar0.ax.tick_params(labelsize=10)

        # Horizontal period label with gray box behind it
        ax[i, 0].text(
            -0.25, 0.5, f"$p_{{random}}$ = {p}", transform=ax[i, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="lightgray",
                edgecolor="gray", alpha=1.0
            )
        )

        # ----- Right column: 1-dimensional homology -----
        hm1 = sns.heatmap(
            d1, ax=ax[i, 1], cmap="plasma", cbar=True,
            cbar_kws={"label": "1-Wasserstein Distance", "shrink": 1.0}
        )
        ax[i, 1].invert_yaxis()

        if i == 0:
            ax[i, 1].set_title("1-Dimensional Homology", fontsize=14, pad=6)

        if i == nrows - 1:
            ax[i, 1].set_xlabel("Lattice Layer", fontsize=12)
        else:
            ax[i, 1].set_xlabel("")

        ax[i, 1].set_ylabel("")

        ax[i, 1].set_xticks(ticks1 + 0.5)
        ax[i, 1].set_xticklabels(ticks1, rotation=90, fontsize=9)
        ax[i, 1].set_yticks(ticks1 + 0.5)
        ax[i, 1].set_yticklabels(ticks1, rotation=0, fontsize=9)

        cbar1 = hm1.collections[0].colorbar
        cbar1.ax.set_ylabel("1-Wasserstein Distance", fontsize=12, rotation=270, labelpad=16)
        cbar1.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0.10, 0.04, 0.98, 0.91])
    if name:
        plt.savefig(name)
    plt.show()

if __name__ == "__main__":
    # This block only runs when the file is executed directly,
    # not when it's imported as a module
    main()