"""
Module Name: VoterModels.py
Description: Contains functions for conducting simulations for variants of
the coevolving voter model on graphs and hypergraphs, as well
as compute topological statistics from these simulations

Author: Jason LaRuez
Date: 2026
"""

# ============================================================================
# IMPORTS
# ============================================================================
import dionysus as d # C++ package with python bindings for persistent homology

import networkx as nx # Network structures
import numpy as np # Numpy arrays and operations
import random # Random sampling for network models
from itertools import combinations, product # For getting different simplices and all combinations of lists

from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt # Plotting
import time # Timing simulations
from tqdm.notebook import tqdm # Allows for real-time progress bar of simulations

import sys
from rbloom import Bloom
import math
import seaborn as sns
import os
import gc # Memory management
import pickle # Takes environment variables and saves them as is
import gzip # Allows for compression of saved files
from joblib import Parallel, delayed # Parallelization functions
import collections

# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================
# Simplicial processing
# ============================================================================

def Get_Simpliciality_SF(filename, processed_filename = None, minDim = 2, maxDim = np.inf):
  """
  Computes the simplicial fraction sigma_SF (proportion of hyperedges,
  above minDim, that are downwardly closed) of a hypergraph, using a
  dynamic-programming pass over edges sorted by ascending size: to
  check if edge e is downward closed it is sufficient to check that
  its |e|-1 sized subsets belong to H and are themselves downward
  closed.

  Parameters
  ----------
  filename : str or dict
    Path to a gzip-pickled hypergraph dict {edge_id: nodes}, or the
    hypergraph dict itself.
  processed_filename : str or None
    If given, path to save the computed output to (gzip-pickled).
  minDim : int
    Minimum edge size to consider. Edges below this size are
    disregarded; edges equal to this size are automatically downward
    closed and are not counted toward sigma_SF.
  maxDim : int or float
    Maximum edge size to consider, to bound computational cost for
    hypergraphs with high-cardinality edges.

  Returns
  -------
  Data : list
    Data[0] : float, sigma_SF, the simplicial fraction.
    Data[1] : collections.Counter, the number of downward-closed edges
              of each size (restricted to minDim < size <= maxDim).
  """
  if type(filename) == dict:
    H = filename
  else:
    # Load data from input filename
    with gzip.open(filename, 'rb') as f:
      try:
        with gzip.open(filename, 'rb') as f:
          H = pickle.load(f)
      except Exception as e:
        print(f"Error loading file: {filename}")
        return -1

  # Create a list of (key, set) tuples sorted by ascending set length
  sorted_items = sorted(H.items(), key=lambda item: len(item[1]), reverse=False)

  # Create a new dictionary with keys 0, 1, 2, ... corresponding to sorted sets
  H = {i: tuple(sorted(item[1])) for i, item in enumerate(sorted_items)}

  # Initialize dict that keeps track of edges in H, as well as whether
  # or not they are downward closed (True or False).
  IsSimplex = {e: True for e in H.values()}

  # Keep track of nodes which are downward closed and total number of
  # hyperedges with |e| > minDim (respectively)
  SimplexCount = 0
  CandidateCount = 0
  E = set()

  # Each timestep represents the addition of a single hyperedge
  for t in tqdm(range(len(H))):
    # To reduce memory overusage we cleanup every so often
    if t > 0 and t % 500000 == 0:
      gc.collect()

    # If the hyperedge is below the minimum size we discard it from consideration.
    # If hyperedge is at minimum size it is automatically a simplex, and so it is
    # disregarded from simpliciality computation
    e = H[t]

    if (len(e) <= minDim) or (len(e) > maxDim) or (e in E):
      continue

    # By default increase simplex and candidate counts by 1.
    SimplexCount += 1; CandidateCount += 1;
    E.add(e)

    # Iterate over the subsets of e of size |e|-1. e is downward closed
    # iff its |e|-1 subsets belong to the hypergraph, and are all downward
    # closed themselves.
    for face in combinations(e, len(e)-1):
      if (face not in IsSimplex) or (not IsSimplex[face]):
        # Since a subset is not a simplex, e is not a simplex
        IsSimplex[e] = False; SimplexCount -= 1
        break

  # Record simplicial fraction, as well as get a dictionary which
  # records the number of edges of each size which are downward closed.
  SimplicialFraction = SimplexCount / CandidateCount
  Lengths = [len(e) for e in IsSimplex if (IsSimplex[e] and len(e) > minDim and len(e) <= maxDim)]
  Lengths = collections.Counter(Lengths)

  # Pickle and save the output
  Data = [SimplicialFraction, Lengths]
  if processed_filename != None:
    with gzip.open(processed_filename, 'wb') as f:
      pickle.dump(Data, f)

    return Data

  else:
    return Data

def Get_Simpliciality_ES(filename, processed_filename=None, maxMemory = 20, stoppingCriteria = None, minDim = 2, maxDim = np.inf):
  """
  Computes the edit simpliciality sigma_ES (distinct from sigma_SF
  and sigma_FES) of a temporally-growing hypergraph at each
  timestep: the number of edges present divided by the number of
  edges present plus the number of missing sub-edges seen so far,
  tracking previously-seen (present and missing) sub-edges with a
  memory-bounded Bloom filter instead of an exact set. Optionally
  stops early once the trajectory appears to have converged.

  Parameters
  ----------
  filename : str or dict
    Path to a gzip-pickled hypergraph dict {edge_id: nodes}, or the
    hypergraph dict itself.
  processed_filename : str or None
    If given, path to save the computed output to (gzip-pickled).
  maxMemory : float
    Maximum memory, in GB, to allocate to the Bloom filter used to
    track missing sub-edges.
  stoppingCriteria : tuple of (int, float) or None
    If given, (window, tol): every `window` timesteps, checks whether
    sigma_ES over the last `window` steps has stayed within `tol` of
    its final value in that window, and if so stops early.
  minDim : int
    Minimum edge size to consider; edges below this size are
    disregarded.
  maxDim : int or float
    Maximum edge size to consider, to bound computational cost.

  Returns
  -------
  float
    sigma_ES at the final computed timestep (or, if stoppingCriteria
    triggers early stopping, the timestep index t at which
    convergence was detected is returned instead). The full
    per-timestep trajectory is written to processed_filename if
    given, rather than returned directly.
  """
  # Load data from input filename
  if type(filename) == dict:
    H = filename
  else:
    # Load data from input filename
    with gzip.open(filename, 'rb') as f:
      try:
        with gzip.open(filename, 'rb') as f:
          H = pickle.load(f)
      except Exception as e:
        print(f"Error loading file: {filename}")
        return -1

  # Get total number of time indices
  T = len(H)

  # Here we determine the largest number of elements we can store
  # in a bloom filter given maxMemory many GB of RAM available.
  error_rate = 0.0001 # Use a smaller error rate for larger datasets
  available_bits = maxMemory * 8 * 10**9

  # Calculate maximum number of elements
  ln2_squared = (math.log(2))**2
  max_elements = int(available_bits * ln2_squared / (-math.log(error_rate)))
  if max_elements > sys.maxsize:
    max_elements = sys.maxsize

  # We iterate over edges in H, counting the maximum number of possible
  # simplices we would need to include in MissingSimplces. Once
  # we hit the max size we set T to be the number of hyperedges we
  # could add before hitting the max size.
  maxFilterSize = 0
  for t in range(T):
    e_len = len(H[t])
    if e_len > maxDim or e_len < minDim:
      continue
    # Calculate the number of potential missing simplices using combinations
    num_potential_missing = 0
    # Iterate over sizes from minDim to e_len - 1
    for r in range(minDim, e_len):
        num_potential_missing += math.comb(e_len, r)

    maxFilterSize += num_potential_missing

    if maxFilterSize >= max_elements:
      maxFilterSize -= num_potential_missing
      T = t; break

  # Ensure maxFilterSize is at least 1 to avoid an error when creating the Bloom filter with size 0.
  # This can happen if T becomes 0 because the first edge is too large or has insufficient dimensions.
  maxFilterSize = max(1, maxFilterSize)
  MissingSimplices = Bloom(maxFilterSize,0.001)
  MissingSimplicesCount = 0
  Edges = set()

  # Initialize arrays for storing simplicialities
  EditSimpliciality = np.zeros(T)

  # Each timestep represents the addition of a single hyperedge
  for t in tqdm(range(T)):
    # To reduce memory overusage we cleanup every so often
    if t > 0 and t % 500000 == 0:
      gc.collect()
    e = tuple(sorted(H[t]))

    # If the edge is smaller than the min size or larger than maxDim we disregard it entirely
    if len(e) < minDim or len(e) > maxDim:
      if t > 0:
        EditSimpliciality[t] = EditSimpliciality[t-1]
      continue

    # Add e to the edges added so far, and check if it was previously labelled missing
    Edges.add(e)
    if e in MissingSimplices:
      MissingSimplicesCount -= 1

    # Having excluded simplices which are too small, we check if the candidate
    # simplices belong to the hypergraph or not.
    for r in range(len(e)-1, minDim-1, -1):
      # combinations(S,r) is from itertools and returns iterator corresponding
      # to all r-subsets of S.
      for face in combinations(e, r):
        # If the subface is not in the hypergraph yet, add it to the missing simplices
        if (face not in Edges) and (face not in MissingSimplices):
          MissingSimplices.add(face)
          MissingSimplicesCount += 1

    # Record number of hyperedges divided by the total number of possible simplices
    denominator = len(Edges) + MissingSimplicesCount
    if denominator > 0:
      EditSimpliciality[t] = len(Edges) / denominator
    else:
      # If denominator is 0, set simpliciality to 0
      EditSimpliciality[t] = 0

    # Here is a stopping criteria which looks at windows of size stoppingCriteria[0],
    # and checks if the maximum deviation from each simpliciality to the simpliciality
    # of the final timestep in the window is less than stoppingCriteria[1]. If so, terminate
    # the for loop.
    if (stoppingCriteria != None) and (t % stoppingCriteria[0] == 0) and (t > 0):
      recentResults = EditSimpliciality[t - stoppingCriteria[0]:t]
      max_diff = max(abs(recentResults[-1] - x) for x in recentResults[:-1])

      if max_diff < stoppingCriteria[1]:
        print(f"Convergence detected at index {t}. Stopping computation.")
        # Pickle and save the output
        with gzip.open(processed_filename, 'wb') as f:
          pickle.dump(EditSimpliciality[:t], f)

        return t

  # Pickle and save the output
  if processed_filename != None:
    with gzip.open(processed_filename, 'wb') as f:
      pickle.dump(EditSimpliciality, f)

  return EditSimpliciality[-1]

def ConstructCliqueComplex(G, k = float('inf')):
    """
    Builds the clique simplicial complex of a networkx graph G, where
    every node, edge, and clique (up to and including size k) is
    encoded as a simplex.

    Parameters
    ----------
    G : networkx.Graph
      The graph to build the clique complex of.
    k : int or float
      Maximum clique size (number of nodes) to include.

    Returns
    -------
    Complex : set of tuple
      The clique complex, as a set of simplices (each a sorted tuple
      of node ids).
    """
    # Find all maximal cliques in G which correspond to the maximal simplices
    Cliques = list(nx.find_cliques(G))
    Complex = set(tuple([v]) for v in G.nodes())
    for e in G.edges:
        Complex.add(e)

    # Iterate across each maximal clique, and include each subset of the clique
    # as a simplex in the complex. The set() structure, with sorted(), avoids the issue of
    # including identical simplices multiple times.
    for clique in Cliques:
        numNodes = len(clique)
        # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
        # all the way down to the 2-subsets (edges) and 1-subsets (nodes)
        # We do sorted() to avoid double creating simplices
        clique = sorted(clique)
        for r in range(min(numNodes, k), 2, -1):
            # combinations(S,r) is from itertools and returns iterator corresponding
            # to all r-subsets of S.
            for face in combinations(clique, r):
                Complex.add(face)

    return (Complex)

def nFaces(Complex, n):
    """
    Filters a simplicial complex down to its n-simplices (simplices
    with n+1 nodes).

    Parameters
    ----------
    Complex : iterable of tuple
      A simplicial complex, as an iterable of simplices.
    n : int
      Simplex dimension to filter for (n >= 0).

    Returns
    -------
    list of tuple
      The simplices of Complex with exactly n+1 nodes.
    """
    # Filter function iterates through the faces of the complex and filters those with n+1 nodes
    return list(filter(lambda face: len(face) == n+1, Complex))

def SimplexCounts(Complex):
    """
    Counts the number of simplices of each size in a simplicial
    complex, and finds the complex's dimension.

    Parameters
    ----------
    Complex : iterable of tuple
      A simplicial complex, as an iterable of simplices.

    Returns
    -------
    Counts : list of int
      Counts[i] is the number of (i)-simplices (i.e. i+1 nodes);
      Counts[0] is the vertex count.
    Dim : int
      The complex's dimension (the largest n with a nonempty n-simplex
      count).
    """
    Counts = [len(nFaces(Complex, 0))]
    Dim = 0
    while True:
        temp = len(nFaces(Complex, Dim + 1))
        if temp > 0:
            Counts.append(temp)
            Dim = Dim + 1
        else:
            break
    return Counts, Dim

def ComputeResults(G, p = 2):
  """
  Computes the Euler characteristic and Betti numbers (b0, b1, b2) of
  the clique complex of a graph (truncated to simplices of size <=
  4), over Z mod p.

  Parameters
  ----------
  G : networkx.Graph
    The graph to compute results for.
  p : int
    Prime modulus for the homology coefficient field Z mod p.

  Returns
  -------
  list
    [b0, b1, b2, Euler, Counts], where b0/b1/b2 are the Betti numbers
    (float), Euler is the Euler characteristic, and Counts is the
    output of SimplexCounts(Complex)[0] (simplex counts by size).
  """
  Complex = ConstructCliqueComplex(G)
  Counts = SimplexCounts(Complex)[0]; Euler = 0
  for i in range(len(Counts)):
    Euler += Counts[i] * (-1)**i

  Simplices = [(list(simplex), len(simplex)-1) for simplex in Complex if len(simplex) <= 4]

  # Create filtration of simplicial complexes using Times
  f = d.Filtration(Simplices)
  # Compute persistent homology of filtration
  m = d.homology_persistence(f, p)

  # PH doesn't return betti numbers, it returns persistence pairs
  # Here we loop through pairs, any time between the birth
  # and death of the pair corresponds to the existence of a hole
  Betti = np.zeros((4,4))
  one = np.ones(4)
  dgms = d.init_diagrams(m,f)
  for i, dgm in enumerate(dgms):
    for p in dgm:
      Betti[i][int(p.birth):int(min(p.death,4))] += one[int(p.birth):int(min(p.death,4))]

  return [Betti[0][-1], Betti[1][-1], Betti[2][-1], Euler, Counts]

def ComputeHGResults(H):
    """
    Computes the Euler characteristic and Betti numbers (b0, b1, b2)
    of the downward-closure complex of a hypergraph (not a clique
    complex — hypergraph complexes are built by downward closure).

    Parameters
    ----------
    H : dict
      Hypergraph as {edge_id: nodes}, where nodes is an iterable of
      node ids for that hyperedge.

    Returns
    -------
    list
      [b0, b1, b2, euler, counts] in the same format as
      ComputeResults: b0/b1/b2 are the Betti numbers (int, counted as
      infinite-persistence bars), euler is the Euler characteristic,
      and counts is simplex counts by size (dims 0-3). Returns
      [0, 0, 0, 0, [0, 0, 0, 0]] if H has no simplices.
    """
    # ── build downward closure ────────────────────────────────────────────────
    simplices = set()
    for edge in H.values():
        nodes = tuple(sorted(edge))
        for size in range(1, len(nodes) + 1):
            for face in combinations(nodes, size):
                simplices.add(face)

    if not simplices:
        return [0, 0, 0, 0, [0, 0, 0, 0]]

    # ── Dionysus filtration: value = dimension ensures valid ordering ─────────
    f = d.Filtration()
    for s in simplices:
        f.append(d.Simplex(list(s), len(s) - 1))
    f.sort()

    # ── persistent homology ───────────────────────────────────────────────────
    m    = d.homology_persistence(f)
    dgms = d.init_diagrams(m, f)

    # Betti numbers: count essential (infinite) bars per dimension
    betti = [0, 0, 0]
    for i, dgm in enumerate(dgms):
        if i >= 3:
            break
        betti[i] = sum(1 for pt in dgm if pt.death == float('inf'))
    b0, b1, b2 = betti

    # ── simplex counts ────────────────────────────────────────────────────────
    counts = [0, 0, 0, 0]
    for s in simplices:
        dim = len(s) - 1
        if 0 <= dim < 4:
            counts[dim] += 1

    euler = counts[0] - counts[1] + counts[2] - counts[3]

    return [b0, b1, b2, euler, counts]

# ============================================================================
# Voter model on Networks
# ============================================================================

def remove_discordant(edge_id, DiscordantIndex, DiscordantEdges):
    """
    Removes edge_id from the discordant-edge list in O(1), via
    swap-and-pop.

    Parameters
    ----------
    edge_id : hashable
      Id of the edge to remove.
    DiscordantIndex : dict
      Maps each discordant edge id to its position in
      DiscordantEdges; updated in place.
    DiscordantEdges : list
      List of discordant edge ids; updated in place.

    Returns
    -------
    None
    """
    pos = DiscordantIndex.pop(edge_id)
    last = DiscordantEdges[-1]
    DiscordantEdges[pos] = last
    DiscordantEdges.pop()
    if DiscordantEdges:  # update index of moved element
        DiscordantIndex[last] = pos

def add_discordant(edge_id, DiscordantIndex, DiscordantEdges):
    """
    Adds edge_id to the discordant-edge list in O(1).

    Parameters
    ----------
    edge_id : hashable
      Id of the edge to add.
    DiscordantIndex : dict
      Maps each discordant edge id to its position in
      DiscordantEdges; updated in place.
    DiscordantEdges : list
      List of discordant edge ids; updated in place.

    Returns
    -------
    None
    """
    DiscordantIndex[edge_id] = len(DiscordantEdges)
    DiscordantEdges.append(edge_id)

def has_disjoint_tuple(DiscordantEdges, E, source_neighbors, target_neighbors):
    """
    Checks whether any discordant edge is fully disjoint from both a
    source and a target node's neighborhoods, i.e. is a valid
    rewire-to-same target that shares no endpoint with either
    neighborhood (used to detect when no valid rewiring partner
    exists).

    Parameters
    ----------
    DiscordantEdges : list
      Discordant edge ids (indices into E) to search among.
    E : list of tuple
      Edge list; E[t] is the (u, v) node pair for edge id t.
    source_neighbors : set
      Neighbor node ids of the source node.
    target_neighbors : set
      Neighbor node ids of the target node.

    Returns
    -------
    bool
      True if at least one edge in DiscordantEdges has both endpoints
      outside source_neighbors and target_neighbors.
    """
    # "any" function stops as soon as a True is found
    return any((E[t][0] not in source_neighbors and E[t][0] not in target_neighbors) and (E[t][1] not in source_neighbors and E[t][1] not in target_neighbors) for t in DiscordantEdges)

def G_RewireToRandomVoter(G, rho, alpha, exit_criteria = np.inf, timing = False):
    """
    Simulates the coevolving (adaptive network) voter model with
    rewire-to-random on a graph: at each step a discordant edge is
    selected uniformly; with probability alpha (per CLAUDE.md, the
    probability of structural rewiring vs. social influence) its
    source node is rewired to a uniformly random new neighbor,
    otherwise the source node adopts its neighbor's opinion.

    Parameters
    ----------
    G : networkx.Graph
      Initial graph; modified in place and returned.
    rho : float
      Initial proportion of nodes with opinion 1.
    alpha : float
      Probability of structural rewiring vs. social influence.
    exit_criteria : int or float
      Maximum number of timesteps to simulate before stopping even if
      discordant edges remain.
    timing : bool
      Whether to print progress information.

    Returns
    -------
    Proportions : ndarray
      Proportion of opinion-1 nodes at each timestep.
    DiscordantCounts : ndarray
      Proportion of discordant edges at each timestep.
    G : networkx.Graph
      The terminal graph.
    """

    if timing == True:
        print("Initializing graph, variables and data structures",flush=True)
        start = time.time()

    numNodes = len(G.nodes())
    # Set |V|*rho many nodes to opinion 1 and the rest to opinion 0
    Opinions = set(np.random.choice(list(G.nodes()), size=round(rho*numNodes), replace=False))
    Opinions = {v: 1 if v in Opinions else 0 for v in G.nodes()}

    # E will be list of edges and we will keep track of indices of edges in E
    E = list(G.edges())
    N = {v:set() for v in G.nodes()}
    for i, e in enumerate(E):
        for v in e:
            N[v].add(i)
    # Generate list of all edges where the connected nodes have differing (discordant) opinions
    DiscordantEdges = [i for i, edge in enumerate(E) if Opinions[edge[0]] != Opinions[edge[1]]]
    DiscordantIndex = {edge_id: pos for pos, edge_id in enumerate(DiscordantEdges)}

    # Initialize list of opinion proportions
    Proportions = [sum(Opinions.values())/numNodes]
    DiscordantCounts = [len(DiscordantEdges) / len(E)]
    timer = 0

    if timing == True:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(Beginning network evolution)",flush=True)
        start = time.time()

    # Main loop of the model. At each step select a discordant edge at random
    # With probability alpha, the (randomly selected) source node is rewired from the target node to a random
    # node with the same opinion as the source node, and is not already connected to the source node
    # Otherwise (with probability 1 - alpha) the source node adopts the opinion of the target node

    while len(DiscordantEdges) > 0 and timer < exit_criteria:
        # Timing print
        if timing and timer % 10000 == 0:
            print("Timer : "+str(timer),flush=True)
        # Proportions from previous time step, and update timer
        timer += 1; Proportions.append(Proportions[-1])

        # Uniformly sample an element from DiscordantEdges, giving the index of a discordant edge in E
        edgeChoice = np.random.randint(len(DiscordantEdges))
        edge_id = DiscordantEdges[edgeChoice]
        edge = E[edge_id]

        # Choose either 0 or 1 to choose which node in the edge is the source and which is the target
        choice = np.random.randint(2)
        source = edge[choice]
        target = edge[(choice + 1) % 2]

        # Rewiring (probability alpha)
        if random.random() < alpha:
            source_neighbors = set(G.neighbors(source))
            target_neighbors = set(G.neighbors(target))

            while True:
                # Randomly select new target by getting index in E
                choice_id = np.random.randint(len(E))
                if choice_id == edge_id:
                    continue
                targetEdge = E[choice_id]
                choice = np.random.randint(2)
                newSource = targetEdge[choice]
                newTarget = targetEdge[(choice + 1) % 2]

                # Check that newTarget is not already connected to source and newSource is
                # not already connected to target, else draw a different edge
                # Since the average degree of a node is low, this should be faster than a deterministic selection
                if (newTarget not in source_neighbors) and (newTarget != source) and (newTarget != target) and (newSource not in target_neighbors) and (newSource != target) and (newSource != source):
                    break

            # Remove edge from G
            G.remove_edge(source, target)
            N[target].remove(edge_id)

            # If the edge is now in harmony, then we remove it from DiscordantEdges
            if Opinions[source] == Opinions[newTarget]:
                remove_discordant(edge_id, DiscordantIndex, DiscordantEdges)
            # Remove targeted edge
            G.remove_edge(newSource, newTarget)
            # Update edges incident to the two rewired nodes (dont need to change source nodes)
            N[target].add(choice_id); N[newTarget].remove(choice_id); N[newTarget].add(edge_id)

            # If the targeted edge was discordant and is now in harmony, remove it from DiscordantEdges
            if Opinions[newSource] != Opinions[newTarget] and Opinions[newSource] == Opinions[target]:
                remove_discordant(choice_id, DiscordantIndex, DiscordantEdges)
            # If the targeted edge was harmonious and is now in discord, add it to DiscordantEdges
            elif Opinions[newSource] == Opinions[newTarget] and Opinions[newSource] != Opinions[target]:
                add_discordant(choice_id, DiscordantIndex, DiscordantEdges)
            # Add edges (source, newTarget) and (newSource, target) to G
            G.add_edge(source, newTarget);
            G.add_edge(newSource, target)
            # Update edges in E
            E[edge_id] = tuple(sorted([source, newTarget]))
            E[choice_id] = tuple(sorted([newSource, target]))

        # Opinion adoption (probability 1 - alpha)
        else:
            # Source node adopts opinion of target node
            Opinions[source] = Opinions[target]
            remove_discordant(edge_id, DiscordantIndex, DiscordantEdges)

            # Either add 1/n (0->1) or subtract 1/n (1->0) to proportion of 1's
            if Opinions[target] == 1:
                Proportions[timer] += 1/numNodes
            else:
                Proportions[timer] -= 1/numNodes

            # Since we have changed the opinion of source node, we need to update whether edges containing
            # source node are discordant or not
            for Id in N[source]:
                # Already handled this case
                if edge_id == Id:
                    continue
                edge = E[Id]
                # If the opinions of the source and its neighbor differ, then they previously
                # were the same and thus not discordant, must add edge to discordant list
                if Opinions[edge[0]] != Opinions[edge[1]]:
                    add_discordant(Id, DiscordantIndex, DiscordantEdges)
                # If the opinions of the source and its neighbor are the same, then they
                # previously were discordant and so we must remove the edge from the discordant list
                else:
                    remove_discordant(Id, DiscordantIndex, DiscordantEdges)

        DiscordantCounts.append(len(DiscordantEdges) / len(E))

    if timing == True:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)

    return np.array(Proportions), np.array(DiscordantCounts), G

def PC_G_RewireRandom(params):
  """
  Worker function for parallel (joblib) sweeps of the graph
  rewire-to-random voter model: builds an initial graph (Erdos-Renyi,
  Barabasi-Albert, or Watts-Strogatz), runs G_RewireToRandomVoter for
  one parameter combination, and pickles the result. Skips the run if
  its output file already exists.

  Parameters
  ----------
  params : tuple
    (n, m, rho, alpha, iteration, graph): node count, edge count,
    initial opinion-1 proportion, rewiring probability, run index,
    and initial graph model ('ER', 'BA', or otherwise Watts-Strogatz
    with rewiring probability 0).

  Returns
  -------
  int
    0 in all cases. Results ([Proportions, DiscordantCounts, G]) are
    written to a filename derived from params rather than returned
    directly.
  """
  # Grab parameter values from params list
  n = params[0]; m = params[1]; rho = params[2]; alpha = params[3]; iteration = params[4]; graph = params[5]

  # Create filename from params
  filename =  'Voter/Random/G_rewire_random_'+str(n)+'_'+str(m)+'_'+str(rho).replace('.','_')+'_'+str(alpha).replace('.','_')+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
    return 0

  if graph == 'ER':
    G = nx.gnm_random_graph(n,m)
  elif graph == 'BA':
    G = nx.barabasi_albert_graph(n,m)
  else:
    G = nx.watts_strogatz_graph(n,m,0)
  Proportions, DiscordantCounts, G = G_RewireToRandomVoter(G, rho, alpha)
  data = [Proportions, DiscordantCounts, G]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def G_RewireToSameVoter(G, rho, alpha, exit_criteria = np.inf, timing = False):
    """
    Simulates the coevolving (adaptive network) voter model with
    rewire-to-same on a graph: at each step a discordant edge is
    selected uniformly; with probability alpha (structural rewiring
    vs. social influence) its source node is rewired to a new
    neighbor sharing the source's opinion (drawn from another
    discordant edge), otherwise the source node adopts its
    neighbor's opinion. Returns early if alpha=1 and no valid
    same-opinion rewiring partner exists.

    Parameters
    ----------
    G : networkx.Graph
      Initial graph; modified in place and returned.
    rho : float
      Initial proportion of nodes with opinion 1.
    alpha : float
      Probability of structural rewiring vs. social influence.
    exit_criteria : int or float
      Maximum number of timesteps to simulate before stopping even if
      discordant edges remain.
    timing : bool
      Whether to print progress information.

    Returns
    -------
    Proportions : ndarray
      Proportion of opinion-1 nodes at each timestep.
    DiscordantCounts : ndarray
      Proportion of discordant edges at each timestep.
    G : networkx.Graph
      The terminal graph.
    """

    if timing == True:
        print("Initializing graph, variables and data structures",flush=True)
        start = time.time()

    numNodes = len(G.nodes())
    # Set |V|*rho many nodes to opinion 1 and the rest to opinion 0
    Opinions = set(np.random.choice(list(G.nodes()), size=round(rho*numNodes), replace=False))
    Opinions = {v: 1 if v in Opinions else 0 for v in G.nodes()}

    E = list(G.edges())
    N = {v: set() for v in G.nodes()}
    for i, e in enumerate(E):
        for v in e:
            N[v].add(i)

    DiscordantEdges = [i for i, edge in enumerate(E) if Opinions[edge[0]] != Opinions[edge[1]]]
    DiscordantIndex = {edge_id: pos for pos, edge_id in enumerate(DiscordantEdges)}

    # Initialize list of opinion proportions
    Proportions = [sum(Opinions.values())/numNodes]
    DiscordantCounts = [len(DiscordantEdges) / len(E)]
    timer = 0

    if timing == True:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(Beginning network evolution",flush=True)
        start = time.time()

    # Main loop of the model. At each step select a discordant edge at random
    # With probability alpha, the (randomly selected) source node is rewired from the target node to a random
    # node with the same opinion as the source node, and is not already connected to the source node
    # Otherwise (with probability 1 - alpha) the source node adopts the opinion of the target node

    while len(DiscordantEdges) > 0 and timer < exit_criteria:
        # Timing print
        if timing and timer % 1000 == 0:
            print("Timer : "+str(timer),flush=True)
        # Proportions from previous time step, and update timer
        timer += 1; Proportions.append(Proportions[-1])

        # Uniformly select a discordant edge
        edgeChoice = np.random.randint(len(DiscordantEdges))
        edge_id = DiscordantEdges[edgeChoice]
        edge = E[edge_id]

        # Choose either 0 or 1 to choose which node in the edge is the source and which is the target
        choice = np.random.randint(2)
        source = edge[choice]
        target = edge[(choice + 1) % 2]

        # Rewiring (probability alpha)
        if random.random() < alpha:
            source_neighbors = set(G.neighbors(source))
            target_neighbors = set(G.neighbors(target))
            # Check if a valid target edge exists
            if not has_disjoint_tuple(DiscordantEdges, E, source_neighbors, target_neighbors):
                DiscordantCounts.append(DiscordantCounts[-1])
                if alpha == 1.0:
                    return np.array(Proportions), np.array(DiscordantCounts), G
                continue

            while True:
                # Randomly select new target
                targetChoice = np.random.randint(len(DiscordantEdges))
                choice_id = DiscordantEdges[targetChoice]
                if choice_id == edge_id:
                    continue
                targetEdge = E[choice_id]
                if Opinions[targetEdge[0]] == Opinions[source]:
                    newTarget = targetEdge[0]; newSource = targetEdge[1]
                else:
                    newTarget = targetEdge[1]; newSource = targetEdge[0]

                # Check that newTarget is not already connected to source and newSource is
                # not already connected to target, else draw a different edge
                # Since the average degree of a node is low, this should be faster than a deterministic selection
                if (newTarget not in source_neighbors) and (newTarget != source) and (newTarget != target) and (newSource not in target_neighbors) and (newSource != target) and (newSource != source):
                    break

            # Remove edge from G
            G.remove_edge(source, target)
            N[target].remove(edge_id)

            # If the edge is now in harmony, then we remove it from DiscordantEdges
            if Opinions[source] == Opinions[newTarget]:
                remove_discordant(edge_id, DiscordantIndex, DiscordantEdges)

            # Remove targeted edge
            G.remove_edge(newSource, newTarget)
            # Update edges incident to the two rewired nodes (dont need to change source nodes)
            N[target].add(choice_id); N[newTarget].remove(choice_id); N[newTarget].add(edge_id)

            # If the targeted edge was discordant and is now in harmony, remove it from DiscordantEdges
            if Opinions[newSource] != Opinions[newTarget] and Opinions[newSource] == Opinions[target]:
                remove_discordant(choice_id, DiscordantIndex, DiscordantEdges)
            # If the targeted edge was harmonious and is now in discord, add it to DiscordantEdges
            elif Opinions[newSource] == Opinions[newTarget] and Opinions[newSource] != Opinions[target]:
                add_discordant(choice_id, DiscordantIndex, DiscordantEdges)
            # Add edges (source, newTarget) and (newSource, target) to G
            G.add_edge(source, newTarget);
            G.add_edge(newSource, target)
            # Update edges in E
            E[edge_id] = tuple(sorted([source, newTarget]))
            E[choice_id] = tuple(sorted([newSource, target]))

        # Opinion adoption (probability 1 - alpha)
        else:
            # Source node adopts opinion of target node
            Opinions[source] = Opinions[target]
            remove_discordant(edge_id, DiscordantIndex, DiscordantEdges)

            # Either add 1/n (0->1) or subtract 1/n (1->0) to proportion of 1's
            if Opinions[target] == 1:
                Proportions[timer] += 1/numNodes
            else:
                Proportions[timer] -= 1/numNodes

            # Since we have changed the opinion of source node, we need to update whether edges containing
            # source node are discordant or not
            for Id in N[source]:
                # Already handled this case
                if edge_id == Id:
                    continue
                edge = E[Id]
                # If the opinions of the source and its neighbor differ, then they previously
                # were the same and thus not discordant, must add edge to discordant list
                if Opinions[edge[0]] != Opinions[edge[1]]:
                    add_discordant(Id, DiscordantIndex, DiscordantEdges)
                # If the opinions of the source and its neighbor are the same, then they
                # previously were discordant and so we must remove the edge from the discordant list
                else:
                    remove_discordant(Id, DiscordantIndex, DiscordantEdges)

        DiscordantCounts.append(len(DiscordantEdges) / len(E))

    if timing == True:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)

    return np.array(Proportions), np.array(DiscordantCounts), G

def PC_G_RewireSame(params):
  """
  Worker function for parallel (joblib) sweeps of the graph
  rewire-to-same voter model: builds an Erdos-Renyi graph, runs
  G_RewireToSameVoter for one parameter combination, and pickles the
  result. Skips the run if its output file already exists.

  Parameters
  ----------
  params : tuple
    (n, m, rho, alpha, iteration): node count, edge count, initial
    opinion-1 proportion, rewiring probability, and run index.

  Returns
  -------
  int
    0 in all cases. Results ([Proportions, DiscordantEdges, G]) are
    written to a filename derived from params rather than returned
    directly.
  """
  # Grab parameter values from params list
  n = params[0]; m = params[1]; rho = params[2]; alpha = params[3]; iteration = params[4];
  # Create filename from params
  filename =  'Voter/Same/G_rewire_same_'+str(n)+'_'+str(m)+'_'+str(rho).replace('.','_')+'_'+str(alpha).replace('.','_')+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
    return 0

  G = nx.gnm_random_graph(n,m)
  Proportions, DiscordantEdges, G = G_RewireToSameVoter(G, rho, alpha)
  data = [Proportions, DiscordantEdges, G]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def G_TriangleRewireSameVoter(G, rho, alpha, gamma, exit_criteria = np.inf, timing = False):
    """
    Simulates the coevolving (adaptive network) voter model with
    triangle-closing rewire-to-same on a graph: at each step a
    discordant edge is selected uniformly; with probability alpha
    (structural rewiring vs. social influence) the edge's source
    node is rewired, and with probability gamma (triangle-closure
    rewiring vs. plain rewire-to-same) the new neighbor is drawn from
    the source's neighbors-of-neighbors (closing a triangle) rather
    than from another discordant edge; otherwise (probability
    1-alpha) the source node adopts its neighbor's opinion.

    Parameters
    ----------
    G : networkx.Graph
      Initial graph; modified in place and returned.
    rho : float
      Initial proportion of nodes with opinion 1.
    alpha : float
      Probability of structural rewiring vs. social influence.
    gamma : float
      Probability of triangle-closure rewiring vs. rewire-to-same,
      given that structural rewiring occurs.
    exit_criteria : int or float
      Maximum number of timesteps to simulate before stopping even if
      discordant edges remain.
    timing : bool
      Whether to print progress information.

    Returns
    -------
    Proportions : ndarray
      Proportion of opinion-1 nodes at each timestep.
    DiscordantCounts : ndarray
      Proportion of discordant edges at each timestep.
    G : networkx.Graph
      The terminal graph.
    """

    if timing == True:
        print("Initializing graph, variables and data structures",flush=True)
        start = time.time()

    numNodes = len(G.nodes())
    # Set |V|*rho many nodes to opinion 1 and the rest to opinion 0
    Opinions = set(np.random.choice(list(G.nodes()), size=round(rho*numNodes), replace=False))
    Opinions = {v: 1 if v in Opinions else 0 for v in G.nodes()}

    # E will be list of edges and we will keep track of indices of edges in E
    E = list(G.edges())
    N = {v:set() for v in G.nodes()}
    for i, e in enumerate(E):
        for v in e:
            N[v].add(i)
    # Generate list of all edges where the connected nodes have differing (discordant) opinions
    DiscordantEdges = [i for i, edge in enumerate(E) if Opinions[edge[0]] != Opinions[edge[1]]]
    DiscordantIndex = {edge_id: pos for pos, edge_id in enumerate(DiscordantEdges)}

    # Initialize list of opinion proportions
    Proportions = [sum(Opinions.values())/numNodes]
    DiscordantCounts = [len(DiscordantEdges) / len(E)]
    timer = 0

    if timing == True:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(Beginning network evolution",flush=True)
        start = time.time()

    # Main loop of the model. At each step select a discordant edge at random
    # With probability alpha, the (randomly selected) source node is rewired from the target node to a random
    # node with the same opinion as the source node, and is not already connected to the source node
    # Otherwise (with probability 1 - alpha) the source node adopts the opinion of the target node

    while len(DiscordantEdges) > 0 and timer < exit_criteria:
        # Timing print
        if timing and timer % 10000 == 0:
            print("Timer : "+str(timer),flush=True)
        # Proportions from previous time step, and update timer
        timer += 1; Proportions.append(Proportions[-1])

        # Uniformly select a discordant edge
        edgeChoice = np.random.randint(len(DiscordantEdges))
        edge_id = DiscordantEdges[edgeChoice]
        edge = E[edge_id]

        # Choose either 0 or 1 to choose which node in the edge is the source and which is the target
        choice = np.random.randint(2)
        source = edge[choice]
        target = edge[(choice + 1) % 2]

        # Rewiring (probability alpha)
        if random.random() < alpha:
            source_neighbors = set(G.neighbors(source))
            target_neighbors = set(G.neighbors(target))

            # Remove edge from G
            G.remove_edge(source, target)
            N[target].remove(edge_id)

            roll = random.random()
            NeighborsOfNeighbors = [None]

            if roll < gamma:
                # Construct the list of neighbors of neighbors of the source node, not including
                # the source nodes neighbors, the source node itself, and any duplicate nodes.
                NeighborsOfNeighbors = list(set(NofN for neighbor in source_neighbors for NofN in G.neighbors(neighbor) if (NofN not in source_neighbors and NofN != source and NofN != target and len(N[NofN]) > 1)))

                # Iterate over valid NofNs checking for valids edges
                while len(NeighborsOfNeighbors) > 0:
                    # Select NofN
                    newTarget = random.choice(NeighborsOfNeighbors)
                    temp = []; counter = 0
                    # Here we check whether the NofN shares 1, or more than 1, neighbor with
                    # the source node. If they share 1, then we need to be careful not to
                    # rewire the shared node as the rewiring would not enforce transitivity
                    for neighbor in G.neighbors(newTarget):
                        if neighbor in source_neighbors:
                            counter += 1
                        else:
                            temp.append(neighbor)
                    # There is at least two common neighbors so we can rewire however we want
                    if counter > 1:
                        cands = list(G.neighbors(newTarget))
                    # There is exactly one common neighbor so we have to be sure not to rewire this neighbor
                    else:
                        cands = temp
                    # Sample and remove invalid cands until a valid node is selected, or no more options exist
                    while cands:
                        idx = random.randrange(len(cands))
                        newSource = cands[idx]
                        if newSource not in target_neighbors and newSource != source and newSource != target:
                            break
                        cands[idx] = cands[-1]
                        cands.pop()
                    # If cands is empty then no valid selection was made
                    if not cands:
                        NeighborsOfNeighbors.remove(newTarget)
                    # Valid selection was made
                    else:
                        # At the end of the gamma branch, after newSource and newTarget are confirmed:
                        choice_id = next(i for i in N[newSource]
                                      if E[i] == tuple(sorted([newSource, newTarget])))
                        break

            # With probability 1-gamma, rewire same selection
            if roll >= gamma or len(NeighborsOfNeighbors) == 0:
                if not has_disjoint_tuple(DiscordantEdges, E, source_neighbors, target_neighbors):
                    DiscordantCounts.append(DiscordantCounts[-1])
                    if alpha == 1.0:
                        return np.array(Proportions), np.array(DiscordantCounts), G
                    G.add_edge(source, target)
                    N[target].add(edge_id)
                    continue

                while True:
                    # Randomly select new target
                    targetChoice = np.random.randint(len(DiscordantEdges))
                    choice_id = DiscordantEdges[targetChoice]
                    if choice_id == edge_id:
                        continue
                    targetEdge = E[choice_id]
                    if Opinions[targetEdge[0]] == Opinions[source]:
                        newTarget = targetEdge[0]; newSource = targetEdge[1]
                    else:
                        newTarget = targetEdge[1]; newSource = targetEdge[0]

                    # Check that newTarget is not already connected to source and newSource is
                    # not already connected to target, else draw a different edge
                    # Since the average degree of a node is low, this should be faster than a deterministic selection
                    if (newTarget not in source_neighbors) and (newTarget != source) and (newTarget != target) and (newSource not in target_neighbors) and (newSource != target) and (newSource != source):
                        break

            # If the edge is now in harmony, then we remove it from DiscordantEdges
            if Opinions[source] == Opinions[newTarget]:
                remove_discordant(edge_id, DiscordantIndex, DiscordantEdges)

            # Remove targeted edge
            G.remove_edge(newSource, newTarget)
            # Update edges incident to the two rewired nodes (dont need to change source nodes)
            N[target].add(choice_id); N[newTarget].remove(choice_id); N[newTarget].add(edge_id)

            # If the targeted edge was discordant and is now in harmony, remove it from DiscordantEdges
            if Opinions[newSource] != Opinions[newTarget] and Opinions[newSource] == Opinions[target]:
                remove_discordant(choice_id, DiscordantIndex, DiscordantEdges)
            # If the targeted edge was harmonious and is now in discord, add it to DiscordantEdges
            elif Opinions[newSource] == Opinions[newTarget] and Opinions[newSource] != Opinions[target]:
                add_discordant(choice_id, DiscordantIndex, DiscordantEdges)
            # Add edges (source, newTarget) and (newSource, target) to G
            G.add_edge(source, newTarget);
            G.add_edge(newSource, target)
            # Update edges in E
            E[edge_id] = tuple(sorted([source, newTarget]))
            E[choice_id] = tuple(sorted([newSource, target]))

        # Opinion adoption (probability 1 - alpha)
        else:
            # Source node adopts opinion of target node
            Opinions[source] = Opinions[target]
            remove_discordant(edge_id, DiscordantIndex, DiscordantEdges)

            # Either add 1/n or subtract 1/n to proportion of 1's
            if Opinions[target] == 1:
                Proportions[timer] += 1/numNodes
            else:
                Proportions[timer] -= 1/numNodes

            # Since we have changed the opinion of source node, we need to update whether edges containing
            # source node are discordant or not
            for Id in N[source]:
                # Already handled this case
                if edge_id == Id:
                    continue
                edge = E[Id]
                # If the opinions of the source and its neighbor differ, then they previously
                # were the same and thus not discordant, must add edge to discordant list
                if Opinions[edge[0]] != Opinions[edge[1]]:
                    add_discordant(Id, DiscordantIndex, DiscordantEdges)
                # If the opinions of the source and its neighbor are the same, then they
                # previously were discordant and so we must remove the edge from the discordant list
                else:
                    remove_discordant(Id, DiscordantIndex, DiscordantEdges)

        DiscordantCounts.append(len(DiscordantEdges) / len(E))

    if timing == True:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)

    return np.array(Proportions), np.array(DiscordantCounts), G

def PC_G_RewireTriangleSame(params):
  """
  Worker function for parallel (joblib) sweeps of the graph
  triangle-closing rewire-to-same voter model: builds an
  Erdos-Renyi graph, runs G_TriangleRewireSameVoter for one
  parameter combination, and pickles the result. Skips the run if
  its output file already exists.

  Parameters
  ----------
  params : tuple
    (n, m, rho, alpha, gamma, iteration): node count, edge count,
    initial opinion-1 proportion, rewiring probability, triangle-
    closure probability, and run index.

  Returns
  -------
  int
    0 in all cases. Results ([Proportions, DiscordantCounts, G]) are
    written to a filename derived from params rather than returned
    directly.
  """
  # Grab parameter values from params list
  n = params[0]; m = params[1]; rho = params[2]; alpha = params[3]; gamma = params[4]; iteration = params[5];
  # Create filename from params
  filename =  'VoterData/G/Triangle/G_rewire_triangle_'+str(n)+'_'+str(m)+'_'+str(rho).replace('.','_')+'_'+str(alpha).replace('.','_')+'_'+str(gamma).replace('.','_')+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
    return 0

  G = nx.gnm_random_graph(n,m)
  Proportions, DiscordantCounts, G = G_TriangleRewireSameVoter(G, rho, alpha, gamma)
  data = [Proportions, DiscordantCounts, G]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

# ============================================================================
# Voter models on hypergraphs
# ============================================================================

def majority_vote(prop):
    """
    Applies majority-vote social influence: an edge's members all
    adopt whichever opinion holds a strict majority, with a coin flip
    to break exact ties.

    Parameters
    ----------
    prop : float
      Proportion of an edge's members currently holding opinion 1.

    Returns
    -------
    int
      1 or 0, the opinion the edge converges to.
    """
    return 1 if prop > 0.5 else (0 if prop < 0.5 else np.random.randint(2))

def proportional_vote(prop):
    """
    Applies proportional-vote social influence: an edge's members
    adopt opinion 1 with probability equal to the current proportion
    of members holding opinion 1 (and opinion 0 otherwise).

    Parameters
    ----------
    prop : float
      Proportion of an edge's members currently holding opinion 1.

    Returns
    -------
    int
      1 or 0, the opinion the edge converges to.
    """
    return 1 if random.random() < prop else 0

def social_influence(H, N, Opinions, Count, DiscordantIndex, DiscordantEdges, edge_id, edgeChoice, vote_function):
    """
    Performs a social-influence step on hyperedge edge_id: applies
    vote_function to determine the edge's converged opinion, updates
    member node opinions accordingly, and updates DiscordantEdges for
    edge_id and any neighboring edges affected by the opinion changes.

    Parameters
    ----------
    H : dict
      Hypergraph as {edge_id: nodes}.
    N : dict
      Maps each node to the set of edge ids it belongs to.
    Opinions : dict
      Maps each node to its current opinion (0 or 1); updated in place.
    Count : dict
      Maps each edge id to its count of opinion-1 members; updated in
      place.
    DiscordantIndex : dict
      Maps each discordant edge id to its position in
      DiscordantEdges; updated in place.
    DiscordantEdges : list
      List of discordant edge ids; updated in place.
    edge_id : hashable
      The edge undergoing social influence.
    edgeChoice : int
      Index of edge_id in DiscordantEdges (unused directly, accepted
      for signature symmetry).
    vote_function : callable
      Voting rule, e.g. majority_vote or proportional_vote.

    Returns
    -------
    changed_count : int
      Number of nodes whose opinion changed.
    Majority : int
      The opinion (0 or 1) the edge converged to.
    """
    edge = H[edge_id]
    # Determine majority class based on voting rule
    Majority = vote_function(Count[edge_id] / len(edge))

    # Update node opinions and record which nodes change
    changed = []
    for v in H[edge_id]:
        if Opinions[v] != Majority:
            Opinions[v] = Majority
            changed.append(v)

    # Update edge proportion and remove from DiscordantEdges
    Count[edge_id] = Majority * len(edge)
    remove_discordant(edge_id, DiscordantIndex, DiscordantEdges)

    # Update neighboring edges based on opinion changes
    if Majority == 1:
        # Nodes changed from 0 to 1
        for v in changed:
            for e in N[v]:
                if e == edge_id:
                    continue
                # Edge was concordant at 0, v changed to 1, add edge to discordant
                if Count[e] == 0:
                    add_discordant(e, DiscordantIndex, DiscordantEdges)
                Count[e] += 1
                # Edge was discordant due to v, v changed to 1, remove edge from discordant
                if Count[e] == len(H[e]):
                    remove_discordant(e, DiscordantIndex, DiscordantEdges)
    else:
        # Nodes changed from 1 to 0
        for v in changed:
            for e in N[v]:
                if e == edge_id:
                    continue
                #  Edge was concordant at 1, v changed to 0, add edge to discordant
                if Count[e] == len(H[e]):
                    add_discordant(e, DiscordantIndex, DiscordantEdges)
                Count[e] -= 1
                if Count[e] == 0:
                    remove_discordant(e, DiscordantIndex, DiscordantEdges)

    return len(changed), Majority


def HG_RewireSame_Voter(H, alpha, rho, voting='Majority',
                         exit_criteria=np.inf, timing=False):
    """
    Simulates the coevolving (adaptive network) voter model with
    rewire-to-same on a hypergraph: at each step a discordant
    hyperedge is selected uniformly; with probability alpha
    (structural rewiring vs. social influence) a minority-opinion
    node in the edge is swapped with an opposite-majority node from
    another edge (rewire-to-same, preserving each edge's size),
    otherwise social influence is applied via vote_function
    (majority_vote or proportional_vote, selected by voting).
    Maintains an O(1) stratified discordant-edge structure (by
    majority opinion) to sample valid rewiring partners efficiently.

    Parameters
    ----------
    H : dict
      Initial hypergraph as {edge_id: nodes}; modified in place and
      returned.
    alpha : float
      Probability of structural rewiring vs. social influence.
    rho : float
      Initial proportion of nodes with opinion 1.
    voting : str
      Voting rule for social influence: 'Majority' or 'Proportional'.
    exit_criteria : int or float
      Maximum number of timesteps to simulate before stopping even if
      discordant edges remain.
    timing : bool
      Whether to print progress information.

    Returns
    -------
    Proportions : ndarray
      Proportion of opinion-1 nodes at each timestep.
    DiscordantCounts : ndarray
      Proportion of discordant edges at each timestep.
    H : dict
      The terminal hypergraph.
    """

    vote_function = majority_vote if voting == 'Majority' else proportional_vote

    if timing:
        print("Initializing hypergraph, variables and data structures", flush=True)
        start = time.time()

    V = set(v for e in H for v in H[e])
    E = list(H.keys())

    N = {v: set() for v in V}
    for e in H:
        for v in H[e]:
            N[v].add(e)

    Opinions = set(np.random.choice(list(V), size=round(rho * len(V)), replace=False))
    Opinions = {v: 1 if v in Opinions else 0 for v in V}

    Count = {e: sum(Opinions[v] for v in H[e]) for e in H}

    def edge_majority(e):
        c = Count[e]; s = len(H[e])
        if c * 2 > s: return 1
        if c * 2 < s: return 0
        return -1  # tied

    # --- O(1) stratified discordant structure ---
    # For each stratum k in {0, 1, -1}:
    #   Strat[k]      : list of edge ids with majority k
    #   StratIndex[k] : dict mapping edge_id -> position in Strat[k]
    # Tied edges (k=-1) are stored only in Strat[-1] and treated as
    # valid targets for either majority at selection time.
    Strat      = {0: [], 1: [], -1: []}
    StratIndex = {0: {}, 1: {}, -1: {}}

    # Master flat list + index for overall discordant tracking
    DiscordantEdges = [e for e in H if 0 < Count[e] < len(H[e])]
    DiscordantIndex = {edge_id: pos for pos, edge_id in enumerate(DiscordantEdges)}

    for e in DiscordantEdges:
        k = edge_majority(e)
        StratIndex[k][e] = len(Strat[k])
        Strat[k].append(e)

    def _strat_add(e, k):
        """Add edge e to stratum k. O(1)."""
        StratIndex[k][e] = len(Strat[k])
        Strat[k].append(e)

    def _strat_remove(e, k):
        """Remove edge e from stratum k using swap-and-pop. O(1)."""
        pos  = StratIndex[k].pop(e)
        last = Strat[k][-1]
        Strat[k][pos] = last
        Strat[k].pop()
        if Strat[k]:
            StratIndex[k][last] = pos

    def add_discordant_strat(e):
        """Add e to flat DiscordantEdges and to its stratum. O(1)."""
        add_discordant(e, DiscordantIndex, DiscordantEdges)
        _strat_add(e, edge_majority(e))

    def remove_discordant_strat(e):
        """Remove e from its stratum and from flat DiscordantEdges. O(1)."""
        _strat_remove(e, edge_majority(e))
        remove_discordant(e, DiscordantIndex, DiscordantEdges)

    def update_strat(e, old_count):
        """
        Re-stratify e after Count[e] changed from old_count.
        Only called when e remains discordant (not concordant).
        O(1).
        """
        old_size = len(H[e])
        if old_count * 2 > old_size:   old_k = 1
        elif old_count * 2 < old_size: old_k = 0
        else:                          old_k = -1
        new_k = edge_majority(e)
        if old_k != new_k:
            _strat_remove(e, old_k)
            _strat_add(e, new_k)

    def sample_opposite(majority, exclude_id):
        """
        Sample uniformly from edges whose majority is opposite to `majority`,
        including tied edges. Returns None if no valid candidate exists.
        O(1) with at most 2 rejection checks.
        """
        opposite = 1 - majority

        # Build combined length for weighted sampling between Strat[opposite]
        # and Strat[-1], without merging the lists.
        n_opp  = len(Strat[opposite])
        n_tied = len(Strat[-1])
        total  = n_opp + n_tied
        if edge_majority(exclude_id) == -1 and n_tied > 0 and total == 1:
            return None
        if total == 0:
            return None

        stuckcounter = 0
        while True:
            stuckcounter += 1;
            if stuckcounter > 1000:
                print("Broken")
            idx = np.random.randint(total)
            if idx < n_opp:
                candidate = Strat[opposite][idx]
            else:
                candidate = Strat[-1][idx - n_opp]
            if candidate != exclude_id:
                return candidate

        return None  # only reached if pool is effectively empty

    Proportions      = [sum(Opinions.values()) / len(V)]
    DiscordantCounts = [len(DiscordantEdges) / len(E)]
    timer = 0

    if timing:
        end = time.time()
        print("Initialization complete, time taken: " + str(end - start) + " seconds",
              flush=True)
        print("Beginning network evolution", flush=True)
        start = time.time()

    while len(DiscordantEdges) > 0 and timer < exit_criteria:
        timer += 1
        Proportions.append(Proportions[-1])

        edgeChoice = np.random.randint(len(DiscordantEdges))
        edge_id    = DiscordantEdges[edgeChoice]
        edge       = H[edge_id]

        Majority = edge_majority(edge_id)
        if Majority == -1:
            Majority = np.random.randint(2)
        Minority = 1 - Majority

        if random.random() < alpha:

            targetEdge_id = sample_opposite(Majority, edge_id)

            if targetEdge_id is None:
                DiscordantCounts.append(len(DiscordantEdges) / len(E))
                continue

            targetEdge = H[targetEdge_id]

            # Minority nodes in source edge (hold opinion Minority)
            source_minority_ids = [
                i for i, v in enumerate(edge)
                if Opinions[v] == Minority and v not in targetEdge
            ]
            # Minority nodes in target edge (hold opinion Majority,
            # since target has opposite majority)
            target_minority_ids = [
                i for i, v in enumerate(targetEdge)
                if Opinions[v] == Majority and v not in edge
            ]

            if not source_minority_ids or not target_minority_ids:
                DiscordantCounts.append(len(DiscordantEdges) / len(E))
                continue

            sourceNode_id = random.choice(source_minority_ids)
            sourceNode    = edge[sourceNode_id]
            targetNode_id = random.choice(target_minority_ids)
            targetNode    = targetEdge[targetNode_id]

            if targetNode == sourceNode:
                DiscordantCounts.append(len(DiscordantEdges) / len(E))
                continue

            # Remove or update strat BEFORE modifying Count
            old_count_source = Count[edge_id]
            old_count_target = Count[targetEdge_id]
            new_source_count = Count[edge_id] + Opinions[targetNode] - Opinions[sourceNode]
            now_concordant_source = (new_source_count == 0 or new_source_count == len(H[edge_id]))

            new_sourceEdge                = list(edge)
            new_sourceEdge[sourceNode_id] = targetNode
            H[edge_id]                    = sorted(new_sourceEdge)

            # Update strat Count changes
            if now_concordant_source:
                remove_discordant_strat(edge_id)
                Count[edge_id] = new_source_count
            else:
                Count[edge_id] = new_source_count
                update_strat(edge_id, old_count_source)

            new_targetEdge                = list(targetEdge)
            new_targetEdge[targetNode_id] = sourceNode
            H[targetEdge_id]              = sorted(new_targetEdge)

            new_target_count = Count[targetEdge_id] + Opinions[sourceNode] - Opinions[targetNode]
            now_concordant_target = (new_target_count == 0 or new_target_count == len(H[targetEdge_id]))

            if now_concordant_target:
                remove_discordant_strat(targetEdge_id)
                Count[targetEdge_id] = new_target_count
            else:
                Count[targetEdge_id] = new_target_count
                update_strat(targetEdge_id, old_count_target)

            N[sourceNode].discard(edge_id)
            N[sourceNode].add(targetEdge_id)
            N[targetNode].add(edge_id)
            N[targetNode].discard(targetEdge_id)

        else:
            # Social influence
            edge_list   = list(H[edge_id])
            Majority_si = vote_function(Count[edge_id] / len(edge_list))

            changed = []
            for v in edge_list:
                if Opinions[v] != Majority_si:
                    Opinions[v] = Majority_si
                    changed.append(v)

            Proportions[timer] += (
                len(changed) / len(V) if Majority_si == 1
                else -len(changed) / len(V)
            )
            remove_discordant_strat(edge_id)
            Count[edge_id] = Majority_si * len(H[edge_id])

            delta = 1 if Majority_si == 1 else -1
            for v in changed:
                for e in N[v]:
                    if e == edge_id:
                        continue
                    old_count_e    = Count[e]
                    was_concordant = (old_count_e == 0
                                      or old_count_e == len(H[e]))
                    new_count_e    = Count[e] + delta
                    now_concordant = (new_count_e == 0
                                      or new_count_e == len(H[e]))

                    if was_concordant and not now_concordant:
                        Count[e] = new_count_e
                        add_discordant_strat(e)
                    elif not was_concordant and now_concordant:
                        remove_discordant_strat(e)
                        Count[e] = new_count_e
                    else:
                        Count[e] = new_count_e
                        update_strat(e, old_count_e)

        DiscordantCounts.append(len(DiscordantEdges) / len(E))

    if timing:
        end = time.time()
        print("Evolution complete, time taken: " + str(end - start) + " seconds",
              flush=True)

    return np.array(Proportions), np.array(DiscordantCounts), H

def HG_RewireTriangleSame_Voter(H, alpha, rho, gamma, voting='Majority',
                                  exit_criteria=np.inf, timing=False):
    """
    Simulates the coevolving (adaptive network) voter model on a
    hypergraph, combining structure-aware transitivity enforcement
    with opinion-aware rewire-to-same as a fallback. With probability
    alpha (structural rewiring vs. social influence), rewiring
    occurs: with probability gamma (triangle-closure rewiring vs.
    rewire-to-same), a triangle-closing swap is attempted, selecting
    nodes purely on structural grounds (no opinion filtering), falling
    back to rewire-to-same if no valid triangle swap exists; with
    probability 1-gamma (or on triangle failure), rewire-to-same is
    used instead, swapping minority-opinion nodes between edges of
    opposite majority to drive consensus. With probability 1-alpha,
    social influence is applied via vote_function (majority_vote or
    proportional_vote, selected by voting).

    Parameters
    ----------
    H : dict
      Initial hypergraph as {edge_id: nodes}; modified in place and
      returned.
    alpha : float
      Probability of structural rewiring vs. social influence.
    rho : float
      Initial proportion of nodes with opinion 1.
    gamma : float
      Probability of triangle-closure rewiring vs. rewire-to-same,
      given that structural rewiring occurs.
    voting : str
      Voting rule for social influence: 'Majority' or 'Proportional'.
    exit_criteria : int or float
      Maximum number of timesteps to simulate before stopping even if
      discordant edges remain.
    timing : bool
      Whether to print progress information.

    Returns
    -------
    Proportions : ndarray
      Proportion of opinion-1 nodes at each timestep.
    DiscordantCounts : ndarray
      Proportion of discordant edges at each timestep.
    H : dict
      The terminal hypergraph.
    """
    vote_function = majority_vote if voting == 'Majority' else proportional_vote

    if timing:
        print("Initializing hypergraph, variables and data structures", flush=True)
        start = time.time()

    V = set(v for e in H for v in H[e])
    E = list(H.keys())

    N = {v: set() for v in V}
    for e in H:
        for v in H[e]:
            N[v].add(e)

    Opinions = set(np.random.choice(list(V), size=round(rho * len(V)), replace=False))
    Opinions = {v: 1 if v in Opinions else 0 for v in V}

    Count = {e: sum(Opinions[v] for v in H[e]) for e in H}

    def edge_majority(e):
        c = Count[e]; s = len(H[e])
        if c * 2 > s: return 1
        if c * 2 < s: return 0
        return -1

    # --- O(1) stratified discordant structure ---
    Strat      = {0: [], 1: [], -1: []}
    StratIndex = {0: {}, 1: {}, -1: {}}

    DiscordantEdges = [e for e in H if 0 < Count[e] < len(H[e])]
    DiscordantIndex = {edge_id: pos for pos, edge_id in enumerate(DiscordantEdges)}

    for e in DiscordantEdges:
        k = edge_majority(e)
        StratIndex[k][e] = len(Strat[k])
        Strat[k].append(e)

    def _strat_add(e, k):
        StratIndex[k][e] = len(Strat[k])
        Strat[k].append(e)

    def _strat_remove(e, k):
        if e not in StratIndex[k]:
            return
        pos = StratIndex[k].pop(e)
        if pos < len(Strat[k]) - 1:
            last = Strat[k][-1]
            Strat[k][pos] = last
            StratIndex[k][last] = pos
        Strat[k].pop()

    def add_discordant_strat(e):
        add_discordant(e, DiscordantIndex, DiscordantEdges)
        _strat_add(e, edge_majority(e))

    def remove_discordant_strat(e):
        _strat_remove(e, edge_majority(e))
        remove_discordant(e, DiscordantIndex, DiscordantEdges)

    def update_strat(e, old_count):
        old_size = len(H[e])
        if old_count * 2 > old_size:   old_k = 1
        elif old_count * 2 < old_size: old_k = 0
        else:                          old_k = -1
        new_k = edge_majority(e)
        if old_k != new_k:
            _strat_remove(e, old_k)
            _strat_add(e, new_k)

    def sample_opposite(majority, exclude_id):
        """O(1) uniform sample from edges with opposite majority."""
        opposite = 1 - majority
        n_opp    = len(Strat[opposite])
        n_tied   = len(Strat[-1])
        total    = n_opp + n_tied
        if total == 0:
            return None
        if total == 1:
            candidate = Strat[opposite][0] if n_opp == 1 else Strat[-1][0]
            return candidate if candidate != exclude_id else None
        stuckcounter = 0
        while True:
            stuckcounter += 1;
            if stuckcounter > 1000:
                print("Broken")
            idx = np.random.randint(total)
            if idx < n_opp:
                candidate = Strat[opposite][idx]
            else:
                candidate = Strat[-1][idx - n_opp]
            if candidate != exclude_id:
                return candidate

    def apply_swap(edge_id, sourceNode_id, sourceNode,
                   targetEdge_id, targetNode_id, targetNode):
        """
        Perform node swap between edge_id and targetEdge_id,
        updating H, Count, Strat and N.
        """
        old_count_source = Count[edge_id]
        old_count_target = Count[targetEdge_id]

        # Update source edge
        new_sourceEdge                = list(H[edge_id])
        new_sourceEdge[sourceNode_id] = targetNode
        H[edge_id]                    = sorted(new_sourceEdge)
        new_source_count              = (old_count_source
                                         + Opinions[targetNode]
                                         - Opinions[sourceNode])

        if new_source_count == 0 or new_source_count == len(H[edge_id]):
            remove_discordant_strat(edge_id)
            Count[edge_id] = new_source_count
        else:
            Count[edge_id] = new_source_count
            update_strat(edge_id, old_count_source)

        # Update target edge
        new_targetEdge                = list(H[targetEdge_id])
        new_targetEdge[targetNode_id] = sourceNode
        H[targetEdge_id]              = sorted(new_targetEdge)
        new_target_count              = (old_count_target
                                         + Opinions[sourceNode]
                                         - Opinions[targetNode])

        was_concordant = (old_count_target == 0
                          or old_count_target == len(H[targetEdge_id]))
        now_concordant = (new_target_count == 0
                          or new_target_count == len(H[targetEdge_id]))

        if was_concordant and not now_concordant:
            Count[targetEdge_id] = new_target_count
            add_discordant_strat(targetEdge_id)
        elif not was_concordant and now_concordant:
            remove_discordant_strat(targetEdge_id)
            Count[targetEdge_id] = new_target_count
        else:
            Count[targetEdge_id] = new_target_count
            update_strat(targetEdge_id, old_count_target)

        # Update node neighbourhoods
        N[sourceNode].discard(edge_id)
        N[sourceNode].add(targetEdge_id)
        N[targetNode].add(edge_id)
        N[targetNode].discard(targetEdge_id)

    Proportions      = [sum(Opinions.values()) / len(V)]
    DiscordantCounts = [len(DiscordantEdges) / len(E)]
    timer = 0

    if timing:
        end = time.time()
        print("Initialization complete, time taken: " + str(end - start) + " seconds",
              flush=True)
        print("Beginning network evolution", flush=True)
        start = time.time()

    while len(DiscordantEdges) > 0 and timer < exit_criteria:
        timer += 1
        Proportions.append(Proportions[-1])

        edgeChoice = np.random.randint(len(DiscordantEdges))
        edge_id    = DiscordantEdges[edgeChoice]
        edge       = H[edge_id]

        Majority = edge_majority(edge_id)
        if Majority == -1:
            Majority = np.random.randint(2)
        Minority = 1 - Majority

        if random.random() < alpha:

            found = False

            # --- Triangle branch (structure-aware, no opinion filtering) ---
            if random.random() < gamma:

                # sourceNode selected purely at random — no opinion constraint
                sourceNode_id = np.random.randint(len(edge))
                sourceNode    = edge[sourceNode_id]

                # Build neighbourhood-of-neighbourhood structure
                Neighbors = set(
                    v for e in N[sourceNode]
                    if e != edge_id
                    for v in H[e]
                    if v != sourceNode
                )
                NeighborEdges = set(
                    e for v in Neighbors
                    for e in N[v]
                    if e not in N[sourceNode]
                )
                NofNs = list(set(
                    v for e in NeighborEdges
                    for v in H[e]
                    if v not in Neighbors and v != sourceNode
                ))

                while NofNs and not found:
                    middleNode_id = np.random.randint(len(NofNs))
                    middleNode    = NofNs[middleNode_id]

                    temp = []; counter = 0
                    for e in N[middleNode]:
                        if e in NeighborEdges:
                            counter += 1
                        else:
                            temp.append(e)
                    cands = list(N[middleNode]) if counter > 1 else temp

                    while cands:
                        cand_idx     = np.random.randint(len(cands))
                        cand_edge_id = cands[cand_idx]
                        cand_edge    = H[cand_edge_id]

                        # Structural validity only: target node must not
                        # already be in source edge, and sourceNode must
                        # not already be in the target edge
                        valid_nodes = [
                            (i, v) for i, v in enumerate(cand_edge)
                            if v != middleNode
                            and v not in edge
                            and sourceNode not in cand_edge
                        ]

                        if valid_nodes:
                            targetEdge_id            = cand_edge_id
                            targetNode_id, targetNode = random.choice(valid_nodes)
                            found = True
                            break
                        else:
                            cands[cand_idx] = cands[-1]
                            cands.pop()

                    if found:
                        break
                    NofNs[middleNode_id] = NofNs[-1]
                    NofNs.pop()

            # --- Fallback: rewire-to-same (opinion-aware) ---
            if not found:
                targetEdge_id = sample_opposite(Majority, edge_id)
                if targetEdge_id is None:
                    DiscordantCounts.append(len(DiscordantEdges) / len(E))
                    if alpha == 1.0:
                        return np.array(Proportions), np.array(DiscordantCounts), H
                    continue

                targetEdge = H[targetEdge_id]

                # Opinion-aware node selection
                source_minority_ids = [
                    i for i, v in enumerate(edge)
                    if Opinions[v] == Minority and v not in targetEdge
                ]
                target_minority_ids = [
                    i for i, v in enumerate(targetEdge)
                    if Opinions[v] == Majority and v not in edge
                ]

                if not source_minority_ids or not target_minority_ids:
                    DiscordantCounts.append(len(DiscordantEdges) / len(E))
                    continue

                sourceNode_id = random.choice(source_minority_ids)
                sourceNode    = edge[sourceNode_id]
                targetNode_id = random.choice(target_minority_ids)
                targetNode    = H[targetEdge_id][targetNode_id]

                if targetNode == sourceNode:
                    DiscordantCounts.append(len(DiscordantEdges) / len(E))
                    continue

            apply_swap(edge_id, sourceNode_id, sourceNode,
                       targetEdge_id, targetNode_id, targetNode)

        else:
            # --- Social influence ---
            edge_list   = list(H[edge_id])
            Majority_si = vote_function(Count[edge_id] / len(edge_list))

            changed = []
            for v in edge_list:
                if Opinions[v] != Majority_si:
                    Opinions[v] = Majority_si
                    changed.append(v)

            Proportions[timer] += (
                len(changed) / len(V) if Majority_si == 1
                else -len(changed) / len(V)
            )
            remove_discordant_strat(edge_id)
            Count[edge_id] = Majority_si * len(H[edge_id])

            delta = 1 if Majority_si == 1 else -1
            for v in changed:
                for e in N[v]:
                    if e == edge_id:
                        continue
                    old_count_e    = Count[e]
                    was_concordant = (old_count_e == 0
                                      or old_count_e == len(H[e]))
                    new_count_e    = Count[e] + delta
                    now_concordant = (new_count_e == 0
                                      or new_count_e == len(H[e]))

                    if was_concordant and not now_concordant:
                        Count[e] = new_count_e
                        add_discordant_strat(e)
                    elif not was_concordant and now_concordant:
                        remove_discordant_strat(e)
                        Count[e] = new_count_e
                    else:
                        Count[e] = new_count_e
                        update_strat(e, old_count_e)

        DiscordantCounts.append(len(DiscordantEdges) / len(E))

    if timing:
        end = time.time()
        print("Evolution complete, time taken: " + str(end - start) + " seconds",
              flush=True)

    return np.array(Proportions), np.array(DiscordantCounts), H

def PC_HG_RewireTriangleSame(params):
  """
  Worker function for parallel (joblib) sweeps of the hypergraph
  triangle-closing rewire-to-same voter model: builds a k-uniform ER
  hypergraph, runs HG_RewireTriangleSame_Voter for one parameter
  combination, and pickles the result. Skips the run if its output
  file already exists.

  Parameters
  ----------
  params : tuple
    (n, m, k, rho, alpha, gamma, voting, iteration): node count,
    hyperedge count, hyperedge size, initial opinion-1 proportion,
    rewiring probability, triangle-closure probability, voting rule,
    and run index.

  Returns
  -------
  int
    0 in all cases. Results ([Proportions, DiscordantCounts, H]) are
    written to a filename derived from params rather than returned
    directly.
  """
  # Grab parameter values from params list
  n = params[0]; m = params[1]; k = params[2]; rho = params[3]; alpha = params[4]; gamma = params[5]; voting = params[6]; iteration = params[7];
  # Create filename from params
  filename =  'VoterData/H/ER/H_rewire_triangle_same_ER_'+str(n)+'_'+str(m)+'_'+str(k)+'_'+str(rho).replace('.','_')+'_'+str(alpha).replace('.','_')+'_'+str(gamma).replace('.','_')+'_'+voting+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
    return 0

  E = set()
  H_ER = {};
  i = 0
  while(i) < m:
    targets = set()
    while len(targets) < k:
      targets.add(np.random.randint(n))
    e = tuple(sorted(list(targets)))
    if e not in E:
      E.add(e)
      H_ER[i] = e
      i += 1

  for e in H_ER:
    H_ER[e] = list(H_ER[e])

  Proportions, DiscordantCounts, H = HG_RewireTriangleSame_Voter(H_ER, alpha, rho, gamma, voting=voting)
  data = [Proportions, DiscordantCounts, H]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def PC_Simplicial_WS(params):
  """
  Worker function for parallel (joblib) sweeps of the hypergraph
  triangle-closing rewire-to-same voter model on a simplicial
  Watts-Strogatz initial hypergraph: builds a WS ring-lattice
  hypergraph (with each node's neighborhood also independently
  gaining extra sub-edges with probability p, increasing
  sigma_SF), runs HG_RewireTriangleSame_Voter for one parameter
  combination, and pickles the result. Skips the run if its output
  file already exists.

  NOTE: this call unpacks 4 return values from
  HG_RewireTriangleSame_Voter, which returns only 3
  (Proportions, DiscordantCounts, H) — as currently written this
  raises a ValueError at runtime. Flagging rather than fixing per
  the current docstring-only task.

  Parameters
  ----------
  params : tuple
    (n, k, rho, alpha, gamma, voting, iteration, p): node count,
    hyperedge size, initial opinion-1 proportion, rewiring
    probability, triangle-closure probability, voting rule, run
    index, and sub-edge inclusion probability.

  Returns
  -------
  int
    0 in all cases. Results are written to a filename derived from
    params rather than returned directly.
  """
  # Grab parameter values from params list
  n = params[0]; k = params[1]; rho = params[2]; alpha = params[3]; gamma = params[4]; voting = params[5]; iteration = params[6]; p = params[7]
  # Create filename from params
  filename =  'Voter/Simplicial/WS_Simplicial_'+str(n)+'_'+str(k)+'_'+str(rho).replace('.','_')+'_'+str(alpha).replace('.','_')+'_'+str(gamma).replace('.','_')+'_'+str(p).replace('.','_')+'_'+voting+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
    return 0

  nodes = range(n)
  E = [tuple(sorted([(v+i) % n for i in range(k)])) for v in nodes]
  for v in nodes:
    for size in range(2,k):
        for comb in combinations(range(v+1,v+k),size-1):
            if random.random() < p:
                E.append(tuple(sorted([v]+[u % n for u in comb])))

  H = {i: e for i, e in enumerate(E)}

  Proportion, DiscordantCount, timer, H = HG_RewireTriangleSame_Voter(H, alpha, rho, gamma, voting=voting, exit_criteria=10000000)
  data = [Proportion, DiscordantCount, timer, H]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def generate_ER_hypergraph(n, m, k):
    """
    Generates a k-uniform Erdos-Renyi hypergraph: m distinct
    size-k hyperedges, each drawn by sampling k nodes uniformly at
    random (without replacement within an edge) from n total nodes.

    Parameters
    ----------
    n : int
      Number of nodes.
    m : int
      Number of hyperedges to generate.
    k : int
      Size of every hyperedge.

    Returns
    -------
    H_ER : dict
      The hyperedges, as {edge_index: nodes (list)}.
    """
    E = set()
    H_ER = {}
    i = 0
    while i < m:
        targets = set()
        while len(targets) < k:
            targets.add(np.random.randint(n))
        e = tuple(sorted(list(targets)))
        if e not in E:
            E.add(e)
            H_ER[i] = list(e)
            i += 1
    return H_ER

def run_single_simulation(params):
    """
    Worker function for parallel (joblib) sweeps of the hypergraph
    triangle-closing rewire-to-same voter model on ER hypergraphs:
    generates a fresh k-uniform ER hypergraph (see
    generate_ER_hypergraph), runs HG_RewireTriangleSame_Voter for one
    parameter combination (with a large exit_criteria cap), and
    pickles only the Proportions/DiscordantCounts trajectories (not
    the terminal hypergraph). Skips the run if its output file
    already exists.

    Parameters
    ----------
    params : tuple
      (k, alpha, gamma, voting, n, m, rho, base_dir): hyperedge size,
      rewiring probability, triangle-closure probability, voting
      rule, node count, hyperedge count, initial opinion-1
      proportion, and output directory.

    Returns
    -------
    tuple
      (success, filename, message): success is a bool, filename the
      output file's basename, and message is 'Already exists',
      'Success', or the exception string on failure.
    """
    k, alpha, gamma, voting, n, m, rho, base_dir = params

    filename = f'ER_{n}_{m}_{k}_{rho}_{alpha}_{gamma}_{voting}.pkl'
    filepath = os.path.join(base_dir, filename)

    # Skip if file already exists
    if os.path.exists(filepath):
        return (True, filename, "Already exists")

    try:
        # Generate hypergraph
        H_ER = generate_ER_hypergraph(n, m, k)

        # Run simulation
        Proportions, DiscordantCounts, H = HG_RewireTriangleSame_Voter(
            H_ER, alpha, rho, gamma, voting=voting, exit_criteria = 100000000
        )

        # Save only Proportions and DiscordantCounts (not H)
        data = {
            'Proportions': Proportions,
            'DiscordantCounts': DiscordantCounts,
            'params': {
                'n': n, 'm': m, 'k': k, 'rho': rho,
                'alpha': alpha, 'gamma': gamma, 'voting': voting
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        return (True, filename, "Success")

    except Exception as e:
        return (False, filename, str(e))

# ============================================================================
# Loading/Plotting
# ============================================================================

def load_single_trajectory_arch(n, m, rho, alpha, gamma, iteration, base_path):
    """
    Loads one graph triangle-rewire trajectory file (see
    PC_G_RewireTriangleSame), from a directory laid out by
    rho/alpha/gamma subfolders, tolerating a missing or corrupted
    file.

    Parameters
    ----------
    n : int
      Node count used to locate the file.
    m : int
      Edge count used to locate the file.
    rho : float
      Initial opinion-1 proportion used to locate the file.
    alpha : float
      Rewiring probability used to locate the file.
    gamma : float
      Triangle-closure probability used to locate the file.
    iteration : int
      Run index used to locate the file.
    base_path : str
      Base directory (containing rho/alpha/gamma subfolders) the file
      is located in.

    Returns
    -------
    Proportions : ndarray or None
      Proportion of opinion-1 nodes at each timestep, or None if the
      file is missing or corrupted.
    DiscordantCounts : ndarray or None
      Proportion of discordant edges at each timestep, or None likewise.
    """
    filename = (
        base_path +
        f'{str(rho).replace(".","_")}/'
        f'{str(alpha).replace(".","_")}/'
        f'{str(gamma).replace(".","_")}/' +
        f'G_rewire_triangle_{n}_{m}_'
        f'{str(rho).replace(".","_")}_'
        f'{str(alpha).replace(".","_")}_'
        f'{str(gamma).replace(".","_")}_'
        f'{iteration}.pkl'
    )

    if not os.path.isfile(filename):
        print(f'File not found: {filename}')
        return None, None

    try:
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)

        # Assuming data format: [Proportions, DiscordantCounts, G]
        Proportions = data[0]
        DiscordantCounts = data[1]

        return Proportions, DiscordantCounts

    except (EOFError, pickle.UnpicklingError) as e:
        print(f'Corrupted file skipped ({type(e).__name__}): {filename}')
        return None, None

def create_graph_arch_figure(graph_data, rho, alpha_values, gamma_values, save_path=None, max_points=500):
    """
    Plots a 3-D "arch" figure for the graph (K=2) triangle-rewire
    voter model: for each (alpha, gamma) combination, plots the
    trajectory (minority proportion, alpha, discordant-edge
    proportion) as a curve in 3-D, colored by gamma, with both the
    minority-opinion trajectory and its mirror (1 - proportion)
    drawn.

    Parameters
    ----------
    graph_data : dict
      Maps (alpha, gamma) -> {'Proportions': ..., 'DiscordantCounts': ...}
      (see load_single_trajectory_arch).
    rho : float
      Initial opinion-1 proportion (used only for context; not
      referenced directly in the plot).
    alpha_values : list of float
      Rewiring probability values to plot curves for.
    gamma_values : list of float
      Triangle-closure probability values to plot curves for,
      colored by a viridis colormap.
    save_path : str or None
      If given, the figure is saved to this path.
    max_points : int
      Unused; accepted for signature compatibility.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
      The created figure and its single 3-D axis.
    """

    gamma_values = [0.2, 0.4, 0.6000000000000001, 0.8, 1.0]
    alpha_values = [0.0, 0.2, 0.4, 0.6000000000000001, 0.8]

    colors = cm.viridis(np.linspace(0, 1, len(gamma_values)))
    color_dict = {gamma: colors[i] for i, gamma in enumerate(gamma_values)}

    # Slightly larger figure
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot curves - OUTER LOOP: alpha (back to front)
    for alpha in alpha_values:
        # INNER LOOP: gamma (colors)
        for gamma in gamma_values:
            key = (alpha, gamma)

            if key not in graph_data:
                continue

            data = graph_data[key]
            Proportions = data['Proportions']
            DiscordantCounts = data['DiscordantCounts']

            # Convert to numpy arrays
            n1_prop = np.array(Proportions)
            DiscordantProp = np.array(DiscordantCounts)

            # Create alpha array
            alpha_array = np.full_like(n1_prop, alpha)

            # Add label only for highest alpha
            add_label = (alpha == alpha_values[-1])

            # Plot original trajectory
            ax.plot(n1_prop, alpha_array, DiscordantProp,
                   color=color_dict[gamma],
                   linewidth=2,
                   alpha=0.8,
                   label=f'γ={gamma:.1f}' if add_label else '')

            # Plot mirrored trajectory
            n1_prop_mirrored = 1.0 - n1_prop
            ax.plot(n1_prop_mirrored, alpha_array, DiscordantProp,
                   color=color_dict[gamma],
                   linewidth=2,
                   alpha=0.8)

    # Formatting
    ax.set_xlabel(r'$\rho_1$ (opinion 1 proportion)', fontsize=14, labelpad=10)
    ax.set_ylabel(r'$\alpha$ (rewiring rate)', fontsize=14, labelpad=10)
    ax.set_zlabel('Discordant Hyperedge Fraction', fontsize=14, labelpad=15, rotation=180)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 0.6])

    ax.set_title(f'Graph Model (K=2), ρ={rho}', fontsize=16, y=.95)
    ax.legend(loc='right', fontsize=13, framealpha=0.9)
    ax.view_init(elev=20, azim=45)

    # Increase left margin to accommodate z-label
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"Saved: {save_path}")

    return fig, ax

def er_raw_fname(base_path, n, m, rho, alpha, gamma, iteration):
    """
    Builds the raw-result filename for one Erdos-Renyi graph
    triangle-rewire voter model run, in a rho/alpha/gamma subfolder
    layout.

    Parameters
    ----------
    base_path : pathlib.Path or str
      Base directory the rho/alpha/gamma subfolders live under.
    n : int
      Node count.
    m : int
      Edge count.
    rho : float
      Initial opinion-1 proportion.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    iteration : int
      Run index.

    Returns
    -------
    str
      The full file path.
    """
    target_dir = (
        base_path
        / str(rho).replace('.', '_')
        / str(alpha).replace('.', '_')
        / str(gamma).replace('.', '_')
    )
    filename = (
        f'G_rewire_triangle_{n}_{m}_'
        f'{str(rho).replace(".", "_")}_'
        f'{str(alpha).replace(".", "_")}_'
        f'{str(gamma).replace(".", "_")}_'
        f'{iteration}.pkl'
    )
    return str(target_dir / filename)

def load_er_proportion(base_path, n, m, alphas, gammas, rho, ai, gi, it):
    """
    Loads one raw Erdos-Renyi graph triangle-rewire result file (see
    er_raw_fname) and extracts the terminal minority opinion
    proportion, tolerating a missing or corrupted file.

    Parameters
    ----------
    base_path : pathlib.Path or str
      Base directory the file is located under.
    n : int
      Node count.
    m : int
      Edge count.
    alphas : ndarray
      Rewiring probability values; alphas[ai] selects the one used.
    gammas : ndarray
      Triangle-closure probability values; gammas[gi] selects the one used.
    rho : float
      Initial opinion-1 proportion.
    ai : int
      Index into alphas.
    gi : int
      Index into gammas.
    it : int
      Run index.

    Returns
    -------
    tuple or None
      (rho, ai, gi, min(p, 1-p)) on success, where p is the terminal
      opinion-1 proportion; None if the file is missing or fails to
      load.
    """
    path = er_raw_fname(base_path, n, m, rho, alphas[ai], gammas[gi], it)
    if not os.path.isfile(path):
        print("File {path} not found")
        return None
    try:
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        p = float(data[0][-1])
        return (rho, ai, gi, min(p, 1.0 - p))
    except Exception:
        print(Exception)
        return None

def hg_raw_fname(base_path, n, m, k, rho, alpha, gamma, voting, iteration):
    """
    Builds the raw-result filename for one hypergraph triangle-rewire
    voter model run, the hypergraph analogue of er_raw_fname.

    Parameters
    ----------
    base_path : str
      Directory the file is located in.
    n : int
      Node count.
    m : int
      Number of hyperedges.
    k : int
      Hyperedge size.
    rho : float
      Initial proportion of opinion 1.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    voting : str
      Voting rule ('Majority' or 'Proportional').
    iteration : int
      Run index.

    Returns
    -------
    str
      The full file path.
    """
    return (
        base_path +
        f'H_rewire_triangle_same_ER_{n}_{m}_{k}_'
        f'{str(rho).replace(".","_")}_'
        f'{str(alpha).replace(".","_")}_'
        f'{str(gamma).replace(".","_")}_'
        f'{voting}_{iteration}.pkl'
    )

def load_hg_er_proportion(base_path, n, m, k, rho, alphas, gammas, voting, ai, gi, iteration):
    """
    Loads one raw hypergraph triangle-rewire result file (see
    hg_raw_fname) and extracts the terminal minority opinion
    proportion and timesteps-to-consensus, the hypergraph analogue of
    load_er_proportion.

    Parameters
    ----------
    base_path : str
      Directory the file is located in.
    n : int
      Node count.
    m : int
      Number of hyperedges.
    k : int
      Hyperedge size.
    rho : float
      Initial proportion of opinion 1.
    alphas : ndarray
      Rewiring probability values; alphas[ai] selects the one used.
    gammas : ndarray
      Triangle-closure probability values; gammas[gi] selects the one used.
    voting : str
      Voting rule ('Majority' or 'Proportional').
    ai : int
      Index into alphas.
    gi : int
      Index into gammas.
    iteration : int
      Run index.

    Returns
    -------
    key : tuple
      (k, voting, ai, gi, iteration), for use as a dict key.
    minority_proportion : float or None
      min(p, 1-p) of the terminal opinion proportion, or None on
      failure.
    timer : int or None
      Timesteps to consensus (or exit), or None on failure.
    """
    alpha = alphas[ai]; gamma = gammas[gi]
    fname = hg_raw_fname(base_path, n, m, k, rho, alpha, gamma, voting, iteration)
    if not os.path.isfile(fname):
        return (k, voting, ai, gi, iteration), None, None
    try:
        with gzip.open(fname, 'rb') as f:
            proportion, _, timer, _ = pickle.load(f)
        term_prop  = float(proportion)
        term_timer = int(timer)
        return (k, voting, ai, gi, iteration), min(1-term_prop, term_prop), term_timer
    except Exception as e:
        print(f'  Error {fname}: {e}')
        return (k, voting, ai, gi, iteration), None, None

def plot_er_terminal(er_means, n, m, rhos, TICK_IDX, XLABELS, YLABELS, save_path=None):
    """
    Plots a 2x2 grid of terminal minority-opinion-proportion heatmaps
    across the (alpha, gamma) grid, one panel per initial opinion-1
    proportion rho, for the Erdos-Renyi graph triangle-rewire model.

    Parameters
    ----------
    er_means : dict
      Maps rho -> 2-D array of mean terminal minority proportion,
      indexed [gamma, alpha].
    n : int
      Node count, shown in the figure title.
    m : int
      Edge count, shown in the figure title.
    rhos : list of float
      Initial opinion-1 proportions to plot, one panel each (up to 4).
    TICK_IDX : ndarray
      Tick positions (into the alpha/gamma grids) for both axes.
    XLABELS : list of str
      Tick labels for the alpha (x) axis.
    YLABELS : list of str
      Tick labels for the gamma (y) axis.
    save_path : str or None
      If given, the figure is saved to this path.

    Returns
    -------
    None
      Displays the figure; saves it to save_path if given.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    for ax, rho in zip(axes.flatten(), rhos):
        data  = er_means[rho]
        label = rf'$\rho_{{\min}}$'

        sns.heatmap(data, ax = ax, cmap = 'RdYlBu_r', mask = np.isnan(data), vmin = 0.0,
            vmax = 0.5, cbar = True, cbar_kws = {'label': label, 'shrink': 0.85},
            xticklabels = False, yticklabels = False, linewidths = 0, rasterized = True)
        ax.invert_yaxis()
        ax.set_xticks(TICK_IDX + 0.5)
        ax.set_xticklabels(XLABELS, rotation=0)
        ax.set_yticks(TICK_IDX + 0.5)
        ax.set_yticklabels(YLABELS, rotation=0)
        ax.set_xlabel(r'$\alpha$ (rewiring rate)')
        ax.set_ylabel(r'$\gamma$ (triangle-closing)')
        ax.set_title(rf'$\rho_0 = {rho}$')
        ax.collections[0].colorbar.ax.tick_params(labelsize=11)
        ax.collections[0].colorbar.set_label(label, size=13)

    fig.suptitle(
        rf'Terminal minority proportion  (ER, $N = {n}$, $M = {m}$)',
        fontsize=15)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()

def proc_graph_fname(rho, alpha, gamma, iteration, PROC_G):
    """
    Builds the processed-TDA filename for one Erdos-Renyi graph
    triangle-rewire voter model run.

    Parameters
    ----------
    rho : float
      Initial opinion-1 proportion.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    iteration : int
      Run index.
    PROC_G : str
      Directory the file is located in.

    Returns
    -------
    str
      The full file path.
    """
    rs = str(rho).replace('.', '_')
    return os.path.join(PROC_G,
        f'tda_rho{rs}_a{alpha:.4f}_g{gamma:.4f}_it{iteration}.pkl.gz')

def compute_and_save_graph(n, m, rho, alpha, gamma, iteration, BASE_G, PROC_G):
    """
    Computes and caches ComputeResults (Betti numbers, Euler
    characteristic, simplex counts of the clique complex) for one
    Erdos-Renyi graph triangle-rewire voter model run, unless already
    cached or its raw result file is missing.

    Parameters
    ----------
    n : int
      Node count.
    m : int
      Edge count.
    rho : float
      Initial opinion-1 proportion.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    iteration : int
      Run index.
    BASE_G : str
      Directory raw result files are located in.
    PROC_G : str
      Directory processed (cached) TDA files are saved to.

    Returns
    -------
    str
      'exists' if already cached, 'missing' if the raw file isn't
      present, 'done' on success, or 'error: <message>' on failure.
      The result dict is written to the cache file, not returned
      directly.
    """
    pfname = proc_graph_fname(rho, alpha, gamma, iteration, PROC_G)
    if os.path.isfile(pfname):
        return 'exists'
    data = load_raw_graph(er_raw_fname(BASE_G, n, m, rho, alpha, gamma, iteration))
    if data is None:
        return 'missing'
    try:
        graph = data[3]
        res  = ComputeResults(graph)
        cnts = res[4] if res[4] is not None else []
        out  = {
            'b0':    res[0],
            'b1':    res[1],
            'b2':    res[2],
            'euler': res[3],
            'N0':    cnts[0] if len(cnts) > 0 else 0,
            'N1':    cnts[1] if len(cnts) > 1 else 0,
            'N2':    cnts[2] if len(cnts) > 2 else 0,
            'N3':    cnts[3] if len(cnts) > 3 else 0,
        }
        with gzip.open(pfname, 'wb') as f:
            pickle.dump(out, f)
        return 'done'
    except Exception as e:
        return f'error: {e}'

def load_one_graph(rho, alphas, gammas, ai, gi, it):
    """
    Loads one cached TDA result (see compute_and_save_graph) for an
    Erdos-Renyi graph triangle-rewire voter model run, tolerating a
    missing or corrupted file.

    NOTE: this calls proc_graph_fname with only 4 positional
    arguments, but proc_graph_fname requires 5 (its final PROC_G
    argument is not supplied here) — as currently written this
    raises a TypeError at runtime. Flagging rather than fixing per
    the current docstring-only task.

    Parameters
    ----------
    rho : float
      Initial opinion-1 proportion.
    alphas : ndarray
      Rewiring probability values; alphas[ai] selects the one used.
    gammas : ndarray
      Triangle-closure probability values; gammas[gi] selects the one used.
    ai : int
      Index into alphas.
    gi : int
      Index into gammas.
    it : int
      Run index.

    Returns
    -------
    tuple or None
      (rho, ai, gi, result_dict) on success (result_dict as returned
      by compute_and_save_graph), or None if the file is missing or
      fails to load.
    """
    pfname = proc_graph_fname(rho, alphas[ai], gammas[gi], it)
    if not os.path.isfile(pfname):
        return None
    try:
        with gzip.open(pfname, 'rb') as f:
            res = pickle.load(f)
        return (rho, ai, gi, res)
    except Exception:
        return None

def gmean(counts_g, sums_g, rho, fld):
    """
    Computes an elementwise mean of one field's accumulated sum over
    its count, for the Erdos-Renyi graph triangle-rewire
    architecture's rho-keyed accumulators.

    Parameters
    ----------
    counts_g : dict
      Accumulator keyed by rho, each an ndarray of sample counts.
    sums_g : dict
      Accumulator keyed by rho, each mapping field names to ndarrays
      of summed values.
    rho : float
      Initial opinion-1 proportion key into counts_g/sums_g.
    fld : str
      Field name key into sums_g[rho].

    Returns
    -------
    ndarray
      Elementwise mean, NaN where the count is 0.
    """
    c = counts_g[rho]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, sums_g[rho][fld] / c, np.nan)

def safe_mean(s, c):
    """
    Computes an elementwise mean s/c, returning NaN wherever the
    count c is 0 rather than raising a divide-by-zero warning/error.

    Parameters
    ----------
    s : ndarray
      Elementwise sum.
    c : ndarray
      Elementwise count.

    Returns
    -------
    ndarray
      s/c elementwise, NaN where c <= 0.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, s / c, np.nan)

def plot_alpha_gamma_heatmap(ax, mat, alphas, gammas, title, cmap='plasma', vmin=None, vmax=None, norm=None):
    """
    Plots a single alpha/gamma heatmap panel with this module's
    standard axis labeling. If norm is not given, vmin/vmax are used
    (auto-computed from the finite values of mat wherever they are
    None) to build a linear Normalize; if norm is given, it takes
    precedence over vmin/vmax.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
      Axis to draw on.
    mat : ndarray
      2-D array to plot, indexed [gamma, alpha].
    alphas : ndarray
      Alpha (rewiring rate) grid values, for the x-axis.
    gammas : ndarray
      Gamma (triangle-closing) grid values, for the y-axis.
    title : str
      Subplot title.
    cmap : str or Colormap
      Colormap to use.
    vmin : float or None
      Lower color-scale bound; ignored if norm is given.
    vmax : float or None
      Upper color-scale bound; ignored if norm is given.
    norm : matplotlib.colors.Normalize or None
      Explicit color normalization (e.g. a LogNorm), overriding vmin/vmax.

    Returns
    -------
    im : matplotlib.collections.QuadMesh
      The plotted mesh, e.g. for use with fig.colorbar.
    """
    if norm is None:
        finite = mat[np.isfinite(mat)]
        if vmin is None and len(finite) > 0:
            vmin = finite.min()
        if vmax is None and len(finite) > 0:
            vmax = finite.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(
        alphas, gammas, mat,
        cmap=cmap, shading='auto',
        norm=norm,
    )
    ax.set_xlabel(r'$\alpha$ (rewiring rate)')
    ax.set_ylabel(r'$\gamma$ (triangle-closing)')
    ax.set_title(title, pad=6)
    ax.set_xlim(alphas[0], alphas[-1])
    ax.set_ylim(gammas[0], gammas[-1])
    return im

def plot_graph_heatmaps_styled(derived_g, CMAP, TICK_IDX, XLABELS, YLABELS, n, m, rho, save_path=None):
    """
    Plots a 3x2 grid of heatmaps across the (alpha, gamma) grid for
    the Erdos-Renyi graph triangle-rewire model's terminal clique
    complex: beta_0, beta_1, N_2, N_3, CR_1 - beta_1, and gamma_1
    (filling efficiency). These quantities are pre-computed by the
    caller and passed in via derived_g; this function only plots them.

    Parameters
    ----------
    derived_g : dict
      Maps rho -> {label: 2-D array}, indexed [gamma, alpha], for
      each of the six panel quantities (pre-computed by the caller).
    CMAP : str or Colormap
      Colormap to use.
    TICK_IDX : ndarray
      Tick positions (into the alpha/gamma grids) for both axes.
    XLABELS : list of str
      Tick labels for the alpha (x) axis.
    YLABELS : list of str
      Tick labels for the gamma (y) axis.
    n : int
      Node count, shown in the figure title.
    m : int
      Edge count, shown in the figure title.
    rho : float
      Initial opinion-1 proportion; selects derived_g[rho] and is
      shown in the figure title.
    save_path : str or None
      If given, the figure is saved to this path.

    Returns
    -------
    None
      Displays the figure; saves it to save_path if given.
    """
    panels = [
        (r'$\beta_0$ (components)',                      derived_g[rho][r'$\beta_0$ (components)']),
        (r'$\beta_1$ (tunnels)',                      derived_g[rho][r'$\beta_1$ (tunnels)']),
        (r'$N_2$ (triangles)',                          derived_g[rho][r'$N_2$ (triangles)']),
        (r'$N_3$ (tetrahedra)',                          derived_g[rho][r'$N_3$ (tetrahedra)']),
        (r'$\mathcal{CR}_1 - \beta_1$ (filled tunnels)',    derived_g[rho][r'$\mathcal{CR}_1 - \beta_1$ (filled tunnels)']),
        (r'$\gamma_1$ (1-filling efficiency)',                     derived_g[rho][r'$\gamma_1$ (1-filling efficiency)']),]

    fig, axes = plt.subplots(3, 2, figsize=(10, 11), constrained_layout=True)

    for ax, (label, data) in zip(axes.flatten(), panels):
        sns.heatmap(data, ax = ax, cmap = CMAP, mask = np.isnan(data), cbar = True, cbar_kws = {'label': label, 'shrink': 0.85},
            xticklabels = False, yticklabels = False, linewidths = 0, rasterized = True)
        ax.invert_yaxis()

        ax.set_xticks(TICK_IDX + 0.5)
        ax.set_xticklabels(XLABELS, rotation=0)
        ax.set_yticks(TICK_IDX + 0.5)
        ax.set_yticklabels(YLABELS, rotation=0)
        ax.set_xlabel(r'$\alpha$ (rewiring rate)')
        ax.set_ylabel(r'$\gamma$ (triangle-closing)')
        ax.set_title(label)

        # match colorbar label font size
        ax.collections[0].colorbar.ax.tick_params(labelsize=9)
        ax.collections[0].colorbar.set_label(label, size=11)

    fig.suptitle(
        rf'Terminal graph complex (ER)'
        rf' --- $N = {n}$, $M = {m}$, $\rho = {rho}$', fontsize=13)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()

def raw_fname_baws(BASE_GRAPH, n, m, model, rho, alpha, gamma, iteration):
    """
    Builds the raw-result filename for one Barabasi-Albert or
    Watts-Strogatz graph triangle-rewire voter model run.

    Parameters
    ----------
    BASE_GRAPH : str
      Base directory the model subfolder lives under.
    n : int
      Node count.
    m : int
      Edge count.
    model : str
      Initial graph model, e.g. 'BA' or 'WS'.
    rho : float
      Initial opinion-1 proportion.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    iteration : int
      Run index.

    Returns
    -------
    str
      The full file path.
    """
    return os.path.join(
        BASE_GRAPH, model,
        f'G_{model}_{n}_{m}_'+str(rho).replace('.','_')+'_'+str(alpha).replace('.','_')+'_'+str(gamma).replace('.','_')+'_'+str(iteration)+'.pkl'
    )

def proc_fname_baws(BASE_GRAPH, model, alpha, gamma, iteration):
    """
    Builds the processed-TDA filename for one Barabasi-Albert or
    Watts-Strogatz graph triangle-rewire voter model run, creating
    its containing directory if needed.

    Parameters
    ----------
    BASE_GRAPH : str
      Base directory the model subfolder lives under.
    model : str
      Initial graph model, e.g. 'BA' or 'WS'.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    iteration : int
      Run index.

    Returns
    -------
    str
      The full file path.
    """
    proc_path = os.path.join(BASE_GRAPH, model, 'proc')
    os.makedirs(proc_path, exist_ok=True)
    return os.path.join(proc_path, f'tda_a{alpha:.4f}_g{gamma:.4f}_it{iteration}.pkl.gz')

def load_raw_graph(path):
    """
    Loads a pickled result file, trying gzip-compressed pickle first
    and falling back to plain pickle, tolerating any failure.

    Parameters
    ----------
    path : str
      Path to the file to load.

    Returns
    -------
    object or None
      The unpickled contents, or None if both load attempts fail.
    """
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def compute_and_save_baws(BASE_GRAPH, n, m, model, rho, alpha, gamma, iteration):
    """
    Computes and caches ComputeResults (Betti numbers, Euler
    characteristic, simplex counts of the clique complex) for one
    Barabasi-Albert or Watts-Strogatz graph triangle-rewire voter
    model run, unless already cached or its raw result file is
    missing.

    Parameters
    ----------
    BASE_GRAPH : str
      Base directory raw and processed files are located under.
    n : int
      Node count.
    m : int
      Edge count.
    model : str
      Initial graph model, e.g. 'BA' or 'WS'.
    rho : float
      Initial opinion-1 proportion.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    iteration : int
      Run index.

    Returns
    -------
    str
      'exists' if already cached, 'missing' if the raw file isn't
      present, 'done' on success, or 'error: <message>' on failure.
      The result dict is written to the cache file, not returned
      directly.
    """
    pfname = proc_fname_baws(BASE_GRAPH, model, alpha, gamma, iteration)
    if os.path.isfile(pfname):
        return 'exists'
    data = load_raw_graph(raw_fname_baws(BASE_GRAPH, n, m, model, rho, alpha, gamma, iteration))
    if data is None:
        return 'missing'
    try:
        graph = data[3]
        res  = ComputeResults(graph)
        cnts = res[4] if res[4] is not None else []
        out  = {
            'b0': res[0], 'b1': res[1], 'b2': res[2], 'euler': res[3], 'N0': cnts[0] if len(cnts) > 0 else 0,
            'N1': cnts[1] if len(cnts) > 1 else 0, 'N2': cnts[2] if len(cnts) > 2 else 0, 'N3': cnts[3] if len(cnts) > 3 else 0}
        with gzip.open(pfname, 'wb') as f:
            pickle.dump(out, f)
        return 'done'
    except Exception as e:
        return f'error: {e}'

def load_one_baws(BASE_GRAPH, alphas, gammas, model, ai, gi, it):
    """
    Loads one cached TDA result (see compute_and_save_baws) for a
    Barabasi-Albert or Watts-Strogatz graph triangle-rewire voter
    model run, tolerating a missing or corrupted file.

    Parameters
    ----------
    BASE_GRAPH : str
      Base directory the file is located under.
    alphas : ndarray
      Rewiring probability values; alphas[ai] selects the one used.
    gammas : ndarray
      Triangle-closure probability values; gammas[gi] selects the one used.
    model : str
      Initial graph model, e.g. 'BA' or 'WS'.
    ai : int
      Index into alphas.
    gi : int
      Index into gammas.
    it : int
      Run index.

    Returns
    -------
    tuple or None
      (model, ai, gi, result_dict) on success (result_dict as
      returned by compute_and_save_baws), or None if the file is
      missing or fails to load.
    """
    pfname = proc_fname_baws(BASE_GRAPH, model, alphas[ai], gammas[gi], it)
    if not os.path.isfile(pfname):
        return None
    try:
        with gzip.open(pfname, 'rb') as f:
            res = pickle.load(f)
        return (model, ai, gi, res)
    except Exception:
        return None

def gmean_baws(counts_baws, sums_baws, model, fld):
    """
    Computes an elementwise mean of one field's accumulated sum over
    its count, for the BA/WS graph triangle-rewire architecture's
    model-keyed accumulators.

    Parameters
    ----------
    counts_baws : dict
      Accumulator keyed by model, each an ndarray of sample counts.
    sums_baws : dict
      Accumulator keyed by model, each mapping field names to
      ndarrays of summed values.
    model : str
      Initial graph model key into counts_baws/sums_baws, e.g. 'BA'
      or 'WS'.
    fld : str
      Field name key into sums_baws[model].

    Returns
    -------
    ndarray
      Elementwise mean, NaN where the count is 0.
    """
    c = counts_baws[model]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, sums_baws[model][fld] / c, np.nan)

def plot_baws_heatmaps(n, m, rho, derived_baws, CMAP, TICK_IDX, XLABELS, YLABELS, model, save_path=None):
    """
    Plots a 3x2 grid of heatmaps across the (alpha, gamma) grid for
    one initial graph model's (BA or WS) terminal clique complex.
    The panel quantities are pre-computed by the caller and passed in
    via derived_baws; this function only plots them.

    Parameters
    ----------
    n : int
      Node count, shown in the figure title.
    m : int
      Edge count, shown in the figure title.
    rho : float
      Initial opinion-1 proportion, shown in the figure title.
    derived_baws : dict
      Maps model -> {label: 2-D array}, indexed [gamma, alpha], for
      each panel quantity (pre-computed by the caller).
    CMAP : str or Colormap
      Colormap to use.
    TICK_IDX : ndarray
      Tick positions (into the alpha/gamma grids) for both axes.
    XLABELS : list of str
      Tick labels for the alpha (x) axis.
    YLABELS : list of str
      Tick labels for the gamma (y) axis.
    model : str
      Initial graph model to plot, e.g. 'BA' or 'WS'; selects
      derived_baws[model] and is shown in the figure title.
    save_path : str or None
      If given, the figure is saved to this path.

    Returns
    -------
    None
      Displays the figure; saves it to save_path if given.
    """
    panels = list(derived_baws[model].items())   # 6 panels

    fig, axes = plt.subplots(3, 2, figsize=(10, 11), constrained_layout=True)

    for ax, (label, data) in zip(axes.flatten(), panels):
        sns.heatmap(data, ax = ax, cmap = CMAP, mask = np.isnan(data), cbar = True,
            cbar_kws = {'label': label, 'shrink': 0.85}, xticklabels = False, yticklabels = False, linewidths = 0, rasterized = True)
        ax.invert_yaxis()
        ax.set_xticks(TICK_IDX + 0.5)
        ax.set_xticklabels(XLABELS, rotation=0)
        ax.set_yticks(TICK_IDX + 0.5)
        ax.set_yticklabels(YLABELS, rotation=0)
        ax.set_xlabel(r'$\alpha$ (rewiring rate)')
        ax.set_ylabel(r'$\gamma$ (triangle-closing)')
        ax.set_title(label)
        ax.collections[0].colorbar.ax.tick_params(labelsize=9)
        ax.collections[0].colorbar.set_label(label, size=11)

    fig.suptitle(
        rf'Terminal graph complex ({model})'
        rf' --- $N = {n}$, $M = {m}$, $\rho = {rho}$',
        fontsize=13,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()

def plot_beta2_comparison(n, m, derived_g, derived_baws, CMAP, TICK_IDX, XLABELS, YLABELS, rho, save_path=None):
    """
    Plots beta_2 (voids) heatmaps across the (alpha, gamma) grid side
    by side for the three initial graph models (ER, WS, BA), sharing
    one color scale for direct comparison.

    Parameters
    ----------
    n : int
      Node count, shown in the figure title.
    m : int
      Edge count, shown in the figure title.
    derived_g : dict
      Maps rho -> {label: 2-D array} for the ER model (see
      plot_graph_heatmaps_styled); must include the beta_2 (voids) key.
    derived_baws : dict
      Maps model ('WS', 'BA') -> {label: 2-D array} (see
      plot_baws_heatmaps); must include the beta_2 (voids) key for each.
    CMAP : str or Colormap
      Colormap to use.
    TICK_IDX : ndarray
      Tick positions (into the alpha/gamma grids) for both axes.
    XLABELS : list of str
      Tick labels for the alpha (x) axis.
    YLABELS : list of str
      Tick labels for the gamma (y) axis.
    rho : float
      Initial opinion-1 proportion; selects derived_g[rho] and is
      shown in the figure title.
    save_path : str or None
      Unused; accepted for signature compatibility (this function
      does not save its figure).

    Returns
    -------
    None
      Displays the figure.
    """
    panels = [
        ('ER (random)', derived_g[rho][r'$\beta_2$ (voids)']),
        ('WS (ring lattice)', derived_baws['WS'][r'$\beta_2$ (voids)']),
        ('BA (preferential attachment)', derived_baws['BA'][r'$\beta_2$ (voids)'])]

    # shared colour range so the three models are directly comparable
    all_vals = np.concatenate([data[~np.isnan(data)].ravel() for _, data in panels])
    vmin, vmax = all_vals.min(), all_vals.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, (model, data) in zip(axes, panels):
        sns.heatmap(data, ax = ax, cmap = CMAP, mask = np.isnan(data), cbar = True,
            cbar_kws = {'label': r'$\beta_2$ (voids)', 'shrink': 0.85}, xticklabels = False,
            yticklabels = False, linewidths = 0, rasterized = True)
        ax.invert_yaxis()
        ax.set_xticks(TICK_IDX + 0.5)
        ax.set_xticklabels(XLABELS, rotation=0)
        ax.set_yticks(TICK_IDX + 0.5)
        ax.set_yticklabels(YLABELS, rotation=0)
        ax.set_xlabel(r'$\alpha$ (rewiring rate)')
        ax.set_ylabel(r'$\gamma$ (triangle-closing)' if ax is axes[0] else '')
        ax.set_title(rf'$\beta_2$ — {model}', fontsize=12)
        ax.collections[0].colorbar.ax.tick_params(labelsize=9)
        ax.collections[0].colorbar.set_label(r'$\beta_2$ (voids)', size=11)

    fig.suptitle(
        rf'$\beta_2$ by initial graph model'
        rf' — $N = {n}$, $M = {m}$, $\rho = {rho}$',
        fontsize=13,
    )

    plt.show()

def load_terminal_one(BASE_GRAPH, n, m, rho, alphas, gammas, model, ai, gi, it):
    """
    Loads one raw BA/WS graph triangle-rewire result file (see
    raw_fname_baws) and extracts terminal minority opinion
    proportion, discordant-edge proportion, and trajectory length,
    tolerating a missing or corrupted file.

    Parameters
    ----------
    BASE_GRAPH : str
      Base directory the file is located under.
    n : int
      Node count.
    m : int
      Edge count.
    rho : float
      Initial opinion-1 proportion.
    alphas : ndarray
      Rewiring probability values; alphas[ai] selects the one used.
    gammas : ndarray
      Triangle-closure probability values; gammas[gi] selects the one used.
    model : str
      Initial graph model, e.g. 'BA' or 'WS'.
    ai : int
      Index into alphas.
    gi : int
      Index into gammas.
    it : int
      Run index.

    Returns
    -------
    tuple or None
      (model, ai, gi, minority, discord, length) on success, where
      minority is min(p, 1-p) of the terminal opinion-1 proportion,
      discord is the terminal discordant-edge proportion, and length
      is the trajectory length; None if the file is missing, fails to
      load, or its contents can't be converted.
    """
    path = raw_fname_baws(BASE_GRAPH, n, m, model, rho, alphas[ai], gammas[gi], it)
    data = load_raw_graph(path)
    if data is None:
        return None
    try:
        prop     = float(data[0])
        discord  = float(data[1])
        length   = float(data[2])
        minority = min(prop, 1.0 - prop)
        return (model, ai, gi, minority, discord, length)
    except Exception:
        return None

def plot_terminal_comparison(n, m, RHO, term_means, TICK_IDX, XLABELS, YLABELS,save_path=None):
    """
    Plots terminal minority-opinion-proportion heatmaps across the
    (alpha, gamma) grid side by side for the WS and BA initial graph
    models.

    Parameters
    ----------
    n : int
      Node count, shown in the figure title.
    m : int
      Edge count, shown in the figure title.
    RHO : float
      Initial opinion-1 proportion, shown in the figure title.
    term_means : dict
      Maps model ('WS', 'BA') -> {r'$\\langle \\rho_{\\min} \\rangle$':
      2-D array}, indexed [gamma, alpha] (see load_terminal_one).
    TICK_IDX : ndarray
      Tick positions (into the alpha/gamma grids) for both axes.
    XLABELS : list of str
      Tick labels for the alpha (x) axis.
    YLABELS : list of str
      Tick labels for the gamma (y) axis.
    save_path : str or None
      Unused; accepted for signature compatibility (this function
      does not save its figure).

    Returns
    -------
    None
      Displays the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    for ax, model in zip(axes, ['WS', 'BA']):
        data  = term_means[model][r'$\langle \rho_{\min} \rangle$']
        if model == 'WS':
          label = rf'$WS$ (ring lattice)'
        elif model == 'BA':
          label = rf'$BA$ (preferential attachment)'

        sns.heatmap(data, ax = ax, cmap = 'RdYlBu_r', mask = np.isnan(data),
            cbar = True, cbar_kws = {'label': rf'$\rho_0$', 'shrink': 0.85},
            xticklabels = False, yticklabels = False, linewidths = 0, rasterized = True)
        ax.invert_yaxis()
        ax.set_xticks(TICK_IDX + 0.5)
        ax.set_xticklabels(XLABELS, rotation=0)
        ax.set_yticks(TICK_IDX + 0.5)
        ax.set_yticklabels(YLABELS, rotation=0)
        ax.set_xlabel(r'$\alpha$ (rewiring rate)')
        ax.set_ylabel(r'$\gamma$ (triangle-closing)')
        ax.set_title(label)
        ax.collections[0].colorbar.ax.tick_params(labelsize=9)
        ax.collections[0].colorbar.set_label(rf'$\rho_{{min}}$', size=11)

    fig.suptitle(
        rf'Terminal minority proportion'
        rf' --- $N = {n}$, $M = {m}$, $\rho = {RHO}$',
        fontsize=13, fontweight='bold')

def raw_fname_HG_ER_arch(data_base, n, m, k, rho, alpha, gamma, voting, iteration):
    """
    Builds the raw-result filename for one hypergraph ER
    triangle-rewire architecture-comparison run.

    Parameters
    ----------
    data_base : str
      Directory the file is located in.
    n : int
      Node count.
    m : int
      Number of hyperedges.
    k : int
      Hyperedge size.
    rho : float
      Initial proportion of opinion 1.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    voting : str
      Voting rule ('Majority' or 'Proportional').
    iteration : int
      Run index.

    Returns
    -------
    str
      The full file path.
    """
    return (
        data_base +
        f'H_rewire_triangle_same_ER_{n}_{m}_{k}_'
        f'{str(rho).replace(".","_")}_'
        f'{str(alpha).replace(".","_")}_'
        f'{str(gamma).replace(".","_")}_'
        f'{voting}_{iteration}.pkl'
    )

def proc_fname_HG_ER_arch(proc_base, n, m, k, rho, alpha, gamma, voting, iteration):
    """
    Builds the processed-TDA filename for one hypergraph ER
    triangle-rewire architecture-comparison run.

    Parameters
    ----------
    proc_base : str
      Directory the file is located in.
    n : int
      Node count.
    m : int
      Number of hyperedges.
    k : int
      Hyperedge size.
    rho : float
      Initial proportion of opinion 1.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    voting : str
      Voting rule ('Majority' or 'Proportional').
    iteration : int
      Run index.

    Returns
    -------
    str
      The full file path.
    """
    return (
        proc_base +
        f'TDA_ER_{n}_{m}_{k}_'
        f'{str(rho).replace(".","_")}_'
        f'{str(alpha).replace(".","_")}_'
        f'{str(gamma).replace(".","_")}_'
        f'{voting}_{iteration}.pkl'
    )

def downward_closure(H, max_dim):
    """
    Builds all simplices in the downward closure of hypergraph H up
    to dimension max_dim.

    Parameters
    ----------
    H : dict
      Hypergraph as {edge_id: nodes}.
    max_dim : int
      Maximum simplex dimension to include.

    Returns
    -------
    simplices : list of set
      One set per dimension 0..max_dim; simplices[dim] holds that
      dimension's simplices, each a sorted tuple of node ids.
    """
    simplices = [set() for _ in range(max_dim + 1)]
    for nodes in H.values():
        nodes = sorted(nodes)
        # include subsets of size 1..(max_dim+1)
        for size in range(1, min(len(nodes) + 1, max_dim + 2)):
            for subset in combinations(nodes, size):
                simplices[size - 1].add(subset)
    return simplices

def compute_tda(H, max_dim=3):
    """
    Builds a dimension-filtered simplicial complex from hypergraph H
    (via downward_closure) and computes its terminal Betti numbers
    and simplex counts, using a dionysus filtration where filtration
    time equals simplex dimension.

    Parameters
    ----------
    H : dict
      Hypergraph as {edge_id: nodes}.
    max_dim : int
      Maximum simplex dimension to include (default 3, i.e. up to
      tetrahedra).

    Returns
    -------
    dict
      Keys 'b0', 'b1', 'b2' (terminal Betti numbers, counted as
      infinite-persistence bars) and 'N0', 'N1', 'N2', 'N3' (simplex
      counts by dimension).
    """
    simplices = downward_closure(H, max_dim)

    # build dionysus filtration: time = dimension
    f = d.Filtration()
    for dim, sset in enumerate(simplices):
        for s in sset:
            f.append(d.Simplex(list(s), float(dim)))
    f.sort()

    # persistent homology
    m    = d.homology_persistence(f)
    dgms = d.init_diagrams(m, f)

    # terminal Betti numbers = count of infinite bars in each diagram
    inf = float('inf')
    b = [0, 0, 0]
    for dim in range(3):
        if dim < len(dgms):
            b[dim] = sum(1 for pt in dgms[dim] if pt.death == inf)

    counts = [len(s) for s in simplices]   # N0, N1, N2, N3
    while len(counts) < 4:
        counts.append(0)

    return {
        'b0': b[0], 'b1': b[1], 'b2': b[2],
        'N0': counts[0], 'N1': counts[1],
        'N2': counts[2], 'N3': counts[3],
    }

def process_and_save_HG_ER_arch(data_base, proc_base, n, m, k, rho, alphas, gammas, voting, ai, gi, iteration, max_dim=3):
    """
    Computes and caches the TDA result (see compute_tda) for one
    hypergraph ER triangle-rewire architecture-comparison run, unless
    already cached or its raw result file is missing.

    Parameters
    ----------
    data_base : str
      Directory raw result files are located in.
    proc_base : str
      Directory processed (cached) TDA files are saved to.
    n : int
      Node count.
    m : int
      Number of hyperedges.
    k : int
      Hyperedge size.
    rho : float
      Initial proportion of opinion 1.
    alphas : ndarray
      Rewiring probability values; alphas[ai] selects the one used.
    gammas : ndarray
      Triangle-closure probability values; gammas[gi] selects the one used.
    voting : str
      Voting rule ('Majority' or 'Proportional').
    ai : int
      Index into alphas.
    gi : int
      Index into gammas.
    iteration : int
      Run index.
    max_dim : int
      Maximum simplex dimension to include, passed to compute_tda.

    Returns
    -------
    None
      Writes the TDA result to a processed-cache file; prints an
      error message on failure. Returns early (no-op) if the cache
      file already exists or the raw file is missing.
    """
    alpha = alphas[ai]; gamma = gammas[gi]
    pfname = proc_fname_HG_ER_arch(proc_base, n, m, k, rho, alpha, gamma, voting, iteration)
    if os.path.isfile(pfname):
        return  # already processed

    rfname = raw_fname_HG_ER_arch(data_base, n, m, k, rho, alpha, gamma, voting, iteration)
    if not os.path.isfile(rfname):
        return

    try:
        with gzip.open(rfname, 'rb') as f:
            _, _, _, H = pickle.load(f)
        result = compute_tda(H, max_dim=max_dim)
        with gzip.open(pfname, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        print(f'  Error [{k}, a={alpha:.2f}, g={gamma:.2f}, {voting}, {iteration}]: {e}')

def load_HG_ER_arch_tda(proc_base, n, m, k, rho, alphas, gammas, voting, ai, gi, iteration):
    """
    Loads one cached TDA result (see process_and_save_HG_ER_arch) for
    a hypergraph ER triangle-rewire architecture-comparison run,
    tolerating a missing or corrupted file.

    Parameters
    ----------
    proc_base : str
      Directory processed (cached) TDA files are located in.
    n : int
      Node count.
    m : int
      Number of hyperedges.
    k : int
      Hyperedge size.
    rho : float
      Initial proportion of opinion 1.
    alphas : ndarray
      Rewiring probability values; alphas[ai] selects the one used.
    gammas : ndarray
      Triangle-closure probability values; gammas[gi] selects the one used.
    voting : str
      Voting rule ('Majority' or 'Proportional').
    ai : int
      Index into alphas.
    gi : int
      Index into gammas.
    iteration : int
      Run index.

    Returns
    -------
    tuple or None
      (k, ai, gi, voting, result_dict) on success (result_dict as
      returned by compute_tda), or None if the file is missing or
      fails to load.
    """
    alpha  = alphas[ai]; gamma = gammas[gi]
    pfname = proc_fname_HG_ER_arch(proc_base, n, m, k, rho, alpha, gamma, voting, iteration)
    if not os.path.isfile(pfname):
        return None
    try:
        with gzip.open(pfname, 'rb') as f:
            res = pickle.load(f)
        return (k, ai, gi, voting, res)
    except Exception:
        return None

def smean_HG_ER_arch(k, v, fld, counts, sums):
    """
    Computes an elementwise mean of one field's accumulated sum over
    its count, for the hypergraph ER triangle-rewire architecture
    comparison's nested-dict accumulators.

    Parameters
    ----------
    k : object
      Hyperedge-size key into counts/sums.
    v : object
      Voting-rule key into counts[k]/sums[k].
    fld : str
      Field name key into sums[k][v].
    counts : dict
      Nested accumulator, counts[k][v] an ndarray of sample counts.
    sums : dict
      Nested accumulator, sums[k][v][fld] an ndarray of summed values.

    Returns
    -------
    ndarray
      Elementwise mean, NaN where the count is 0.
    """
    c = counts[k][v]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, sums[k][v][fld] / c, np.nan)

def load_single_file(filepath):
    """Load a single pickle file"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_all_HG_arch_data_parallel(base_dir, n=5000, m=10000, rho=0.5):
    """Load all simulation data in parallel"""

    k_values = [3,4]
    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    gamma_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    voting_values = ['Majority','Proportional']

    # Build list of all filepaths
    filepaths = []
    params_list = []

    for k, alpha, gamma, voting in product(k_values, alpha_values, gamma_values, voting_values):
        filename = f'ER_{n}_{m}_{k}_{rho}_{alpha}_{gamma}_{voting}.pkl'
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            filepaths.append(filepath)
            params_list.append((k, alpha, gamma, voting))

    print(f"Loading {len(filepaths)} files in parallel...")

    # Load in parallel
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(load_single_file)(fp) for fp in filepaths
    )

    # Organize into dictionary
    data_dict = {}
    for params, result in zip(params_list, results):
        if result is not None:
            k, alpha, gamma, voting = params
            key = (k, alpha, gamma, voting)
            data_dict[key] = result

    print(f"Successfully loaded {len(data_dict)} datasets")
    return data_dict

def create_HG_arch_figure_pair(all_data, k_value, save_path=None):
    """
    Two-panel 3D arch plot for a fixed K: Majority (left) and Proportional
    (right), with a single shared legend.
    """

    gamma_values  = [0.2, 0.4, 0.6, 0.8, 1.0]
    alpha_values  = [0.0, 0.2, 0.4, 0.6, 0.8]
    voting_rules  = ['Majority', 'Proportional']

    colors     = cm.viridis(np.linspace(0, 1, len(gamma_values)))
    color_dict = {gamma: colors[i] for i, gamma in enumerate(gamma_values)}

    fig = plt.figure(figsize=(20, 9))

    for panel_idx, voting_rule in enumerate(voting_rules):
        ax = fig.add_subplot(1, 2, panel_idx + 1, projection='3d')

        for alpha in alpha_values:
            for gamma in gamma_values:
                key = (k_value, alpha, gamma, voting_rule)
                if key not in all_data:
                    continue

                data             = all_data[key]
                Proportions      = data['Proportions']
                DiscordantCounts = data['DiscordantCounts']

                n1_prop     = np.array(Proportions)
                alpha_array = np.full_like(n1_prop, alpha)

                ax.plot(n1_prop, alpha_array, DiscordantCounts,
                        color=color_dict[gamma],
                        linewidth=2, alpha=0.8)

                # mirrored branch
                ax.plot(1.0 - n1_prop, alpha_array, DiscordantCounts,
                        color=color_dict[gamma],
                        linewidth=2, alpha=0.8)

        ax.set_xlabel(r'$\rho$ (minority proportion)', fontsize=18, labelpad=10)
        ax.set_ylabel(r'$\alpha$ (rewiring rate)', fontsize=18, labelpad=10)
        ax.set_zlabel(r'Proportion of Discordant Edges', fontsize=16, labelpad=10, rotation=180)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

        ax.set_title(f'{voting_rule} Voting', fontsize=20)#, pad=1)
        ax.view_init(elev=20, azim=45)

    # ── shared legend, built manually from the colour map ─────────────────────
    legend_handles = [
        Line2D([0], [0], color=color_dict[g], linewidth=2.5,
               label=f'γ={g:.1f}')
        for g in gamma_values
    ]
    fig.legend(handles=legend_handles,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.10),
               ncol=len(gamma_values),
               fontsize=15,
               framealpha=0.9)

    fig.suptitle(f'K={k_value}', fontsize=22, y=0.9)

    plt.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.14,
                        wspace=0.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"Saved: {save_path}")

    return fig

def raw_fname_simplicial_WS(base_s, n, k, rho, psub, alpha, gamma, voting, iteration):
    """
    Builds the raw-result filename for one simplicial Watts-Strogatz
    hypergraph voter model run.

    Parameters
    ----------
    base_s : str
      Directory the file is located in.
    n : int
      Node count.
    k : int
      Hyperedge size.
    rho : float
      Initial proportion of opinion 1.
    psub : float
      Sub-edge inclusion probability.
    alpha : float
      Rewiring probability.
    gamma : float
      Triangle-closure probability.
    voting : str
      Voting rule ('Majority' or 'Proportional').
    iteration : int
      Run index.

    Returns
    -------
    str
      The full file path.
    """
    return os.path.join(base_s,
        f'WS_Simplicial_{n}_{k}_'
        f'{str(rho).replace(".","_")}_'
        f'{str(alpha).replace(".","_")}_'
        f'{str(gamma).replace(".","_")}_'
        f'{str(psub).replace(".","_")}_'
        f'{voting}_{iteration}.pkl')

def proc_fname_simplicial_WS(proc_s, n, k, psub, ai, gi, iteration):
    """
    Builds the processed-TDA filename for one simplicial
    Watts-Strogatz hypergraph voter model run.

    Parameters
    ----------
    proc_s : str
      Directory the file is located in.
    n : int
      Node count.
    k : int
      Hyperedge size.
    psub : float
      Sub-edge inclusion probability.
    ai : int
      Index into the rewiring-probability grid.
    gi : int
      Index into the triangle-closure-probability grid.
    iteration : int
      Run index.

    Returns
    -------
    str
      The full file path.
    """
    return os.path.join(proc_s,
        f'proc_{n}_{k}_psub{str(psub).replace(".","_")}_'
        f'a{ai}_g{gi}_it{iteration}.pkl.gz')

def compute_and_save_HG(base_s, proc_s, n, k, rho, alphas, gammas, voting, psub, ai, gi, iteration):
    """
    Computes and caches sigma_SF, sigma_ES, and Betti/simplex-count
    TDA results (via ComputeHGResults, downward closure) for one
    simplicial Watts-Strogatz hypergraph voter model run, unless
    already cached or its raw result file is missing.

    Parameters
    ----------
    base_s : str
      Directory raw result files are located in.
    proc_s : str
      Directory processed (cached) TDA files are saved to.
    n : int
      Node count.
    k : int
      Hyperedge size.
    rho : float
      Initial proportion of opinion 1.
    alphas : ndarray
      Rewiring probability values; alphas[ai] selects the one used.
    gammas : ndarray
      Triangle-closure probability values; gammas[gi] selects the one used.
    voting : str
      Voting rule ('Majority' or 'Proportional').
    psub : float
      Sub-edge inclusion probability.
    ai : int
      Index into alphas.
    gi : int
      Index into gammas.
    iteration : int
      Run index.

    Returns
    -------
    str
      'exists' if already cached, 'missing' if the raw file isn't
      present, 'done' on success, or 'error: <message>' on failure.
      The result dict is written to the cache file, not returned
      directly.
    """
    pfname = proc_fname_simplicial_WS(proc_s, n, k, psub, ai, gi, iteration)
    if os.path.isfile(pfname):
        return 'exists'
    rpath = raw_fname_simplicial_WS(base_s, n, k, rho, psub, alphas[ai], gammas[gi], voting, iteration)
    if not os.path.isfile(rpath):
        return 'missing'
    try:
        with gzip.open(rpath, 'rb') as f:
            data = pickle.load(f)

        proportion = float(data[0])
        H          = data[3]

        tda  = ComputeHGResults(H)          # ← downward closure, not clique
        cnts = tda[4] if tda[4] is not None else []

        sf = float(Get_Simpliciality_SF(H, minDim=2, maxDim=np.inf)[0])
        es = float(Get_Simpliciality_ES(H, minDim=2, maxDim=np.inf))

        out = {
            'proportion': proportion,
            'b0':         tda[0],
            'b1':         tda[1],
            'b2':         tda[2],
            'euler':      tda[3],
            'N0':         cnts[0] if len(cnts) > 0 else 0,
            'N1':         cnts[1] if len(cnts) > 1 else 0,
            'N2':         cnts[2] if len(cnts) > 2 else 0,
            'N3':         cnts[3] if len(cnts) > 3 else 0,
            'sf':         sf,
            'es':         es}
        with gzip.open(pfname, 'wb') as f:
            pickle.dump(out, f)
        return 'done'
    except Exception as e:
        return f'error: {e}'

def load_simplicial_WS_tda(proc_s, n, k, psub, ai, gi, iteration):
    """
    Loads one cached TDA result (see compute_and_save_HG) for a
    simplicial Watts-Strogatz hypergraph voter model run, tolerating
    a missing or corrupted file.

    Parameters
    ----------
    proc_s : str
      Directory processed (cached) TDA files are located in.
    n : int
      Node count.
    k : int
      Hyperedge size.
    psub : float
      Sub-edge inclusion probability.
    ai : int
      Index into the rewiring-probability grid.
    gi : int
      Index into the triangle-closure-probability grid.
    iteration : int
      Run index.

    Returns
    -------
    tuple or None
      (k, psub, ai, gi, result_dict) on success, or None if the file
      is missing or fails to load.
    """
    pfname = proc_fname_simplicial_WS(proc_s, n, k, psub, ai, gi, iteration)
    if not os.path.isfile(pfname):
        return None
    try:
        with gzip.open(pfname, 'rb') as f:
            res = pickle.load(f)
        return (k, psub, ai, gi, res)
    except Exception:
        return None

def smean_simplicial_WS(k, psub, fld, counts, sums):
    """
    Computes an elementwise mean of one field's accumulated sum over
    its count, for the simplicial Watts-Strogatz hypergraph
    accumulators keyed by (k, psub) tuples.

    Parameters
    ----------
    k : object
      Hyperedge-size component of the (k, psub) key into counts/sums.
    psub : object
      Sub-edge-probability component of the (k, psub) key.
    fld : str
      Field name key into sums[(k, psub)].
    counts : dict
      Accumulator keyed by (k, psub) tuples, each an ndarray of
      sample counts.
    sums : dict
      Accumulator keyed by (k, psub) tuples, each mapping field names
      to ndarrays of summed values.

    Returns
    -------
    ndarray
      Elementwise mean, NaN where the count is 0.
    """
    c = counts[(k, psub)]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, sums[(k, psub)][fld] / c, np.nan)

def plot_simplicial_heatmaps(n, rho, derived, CMAP, XLABELS, YLABELS, TICK_A, TICK_G,
                             k, psub, save_path=None):
    """
    Plots a grid of heatmaps across the (alpha, gamma) grid for one
    (hyperedge size, sub-edge probability) combination of the
    simplicial Watts-Strogatz hypergraph voter model (e.g. sigma_SF,
    sigma_ES, beta_1, and terminal minority proportion). The panel
    quantities are pre-computed by the caller and passed in via
    derived; this function only plots them.

    Parameters
    ----------
    n : int
      Node count, shown in the figure title.
    rho : float
      Initial opinion-1 proportion, shown in the figure title.
    derived : dict
      Maps (k, psub) -> {label: 2-D array}, indexed [gamma, alpha],
      for each panel quantity (pre-computed by the caller).
    CMAP : str or Colormap
      Colormap to use.
    XLABELS : list of str
      Tick labels for the alpha (x) axis.
    YLABELS : list of str
      Tick labels for the gamma (y) axis.
    TICK_A : ndarray
      Tick positions into the alpha grid.
    TICK_G : ndarray
      Tick positions into the gamma grid.
    k : int
      Hyperedge size; selects derived[(k, psub)] and is shown in the
      figure title.
    psub : float
      Sub-edge inclusion probability; selects derived[(k, psub)] and
      is shown in the figure title.
    save_path : str or None
      If given, the figure is saved to this path.

    Returns
    -------
    None
      Displays the figure; saves it to save_path if given.
    """
    panels   = list(derived[(k, psub)].items())
    n_panels = len(panels)
    n_cols, n_rows = 2, (n_panels + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8, 3.5 * n_rows),
                             constrained_layout=True)
    for ax, (label, data) in zip(axes.flatten(), panels):
        sns.heatmap(data, ax=ax, cmap=CMAP, mask=np.isnan(data),
                    cbar=True, cbar_kws={'label': label, 'shrink': 0.85},
                    xticklabels=False, yticklabels=False,
                    linewidths=0, rasterized=True)
        ax.invert_yaxis()
        ax.set_xticks(TICK_A + 0.5); ax.set_xticklabels(XLABELS, rotation=0)
        ax.set_yticks(TICK_G + 0.5); ax.set_yticklabels(YLABELS, rotation=0)
        ax.set_xlabel(r'$\alpha$ (rewiring rate)')
        ax.set_ylabel(r'$\gamma$ (triangle-closing)')
        ax.set_title(label)
        ax.collections[0].colorbar.ax.tick_params(labelsize=9)
        ax.collections[0].colorbar.set_label(label, size=11)
    for ax in axes.flatten()[n_panels:]:
        ax.set_visible(False)
    fig.suptitle(
        rf'Simplicial WS voter model  ($K={k}$, $p_{{\mathrm{{sub}}}}={psub}$,'
        rf' $N={n}$, $\rho={rho}$, Majority vote)',
        fontsize=13)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()