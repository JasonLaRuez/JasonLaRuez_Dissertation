"""
Module Name: NetworkModels.py
Description: Contains functions for generating networks,
computing topological quantities from these models,
and construct plots

Author: Jason LaRuez
Date: 2026
"""

# ============================================================================
# IMPORTS
# ============================================================================
import networkx as nx # Network structures
import numpy as np # Numpy arrays and operations
import random # Random sampling for network models
from itertools import combinations # For getting different simplices and all combinations of lists

import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.pyplot as plt # Plotting
import time # Timing simulations
from tqdm.notebook import tqdm # Allows for real-time progress bar of simulations

import gc # Memory management
import pickle # Takes environment variables and saves them as is
import gzip # Allows for compression of saved files

import os # Iterating over directories for files
import math 
from math import comb
import bisect
import collections

import dionysus as d # C++ package with python bindings for persistent homology

# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================
# Generative Models
# ============================================================================

def ClusterCoeffUpdate(G,nodes,clusterC,clust_contrib,clust_sum,timer):
    """
    Incrementally updates the running average clustering coefficient
    after a graph change affecting the given nodes: since adding or
    removing edges only affects the local clustering of the changed
    nodes and their common neighbors, only those nodes are recomputed
    (via nx.clustering) rather than the whole graph.

    Parameters
    ----------
    G : networkx.Graph
      The current graph, after the change has already been applied.
    nodes : iterable
      Nodes directly involved in the change (e.g. edge endpoints);
      their neighbors are also recomputed.
    clusterC : ndarray
      Per-timestep average clustering coefficient time series;
      clusterC[timer] is updated in place.
    clust_contrib : dict
      Maps each node to its current clustering coefficient
      contribution; updated in place for affected nodes.
    clust_sum : float
      Running sum of all nodes' clustering coefficients.
    timer : int
      Current timestep index.

    Returns
    -------
    float
      The updated clust_sum.
    """
    affected = set(nodes)
    for u in nodes:
        affected.update(set(G.neighbors(u)))

    # Subtract old clustering contributions of affected nodes
    clust_sum -= sum(clust_contrib[w] for w in affected)

    # Recompute clustering for affected nodes only via nx.clustering
    new_clusts = nx.clustering(G, nodes=affected)
    for w, c in new_clusts.items():
        clust_contrib[w] = c
        clust_sum += c

    clusterC[timer] = clust_sum / len(clust_contrib)
    return clust_sum


def PH_ErdosRenyi(n, p = 1, timing = False, maxDim = np.inf):
  """
  Simulates an evolving Erdos-Renyi graph: starting from n isolated
  nodes, every possible edge is added in a uniformly random order
  (stopping once edge density p is reached), computing persistent
  homology of the clique complex, cycle rank, and average clustering
  coefficient as it grows.

  Parameters
  ----------
  n : int
    Number of nodes.
  p : float
    Terminal edge density (0 <= p <= 1) at which to stop adding edges.
  timing : bool
    Whether to display a progress bar.
  maxDim : int or float
    Maximum simplex dimension to track.

  Returns
  -------
  Betti : ndarray, shape (4, timer+1)
    Betti[k][t] is the k-th Betti number at timestep t (dim 3 is
    truncated, since simplices above dim 3 are not tracked).
  SimplexCounts : ndarray
    Simplex counts by size at each timestep.
  Euler : ndarray, shape (timer+1,)
    Euler characteristic at each timestep.
  cycleR : list, length timer+1
    CR_1 (= N_1 - N_0 + beta_0) at each timestep.
  clusterC : ndarray, length timer+1
    Average clustering coefficient at each timestep.
  """

  if timing:
    print("(1/5) Initializing graph, variables and data structures",flush=True)
    start = time.time()

  # Construct empty graph and shuffled list of every possible edge
  G = nx.empty_graph(n)
  E = [(i,j) for i in range(n) for j in range(i+1,n)]
  random.shuffle(E)

  # Initialize array of simplices and times they were added to complex
  Times = []

  # Add vertex simplices
  for i in range(n):
    Times.append(([i], 0))

  # Intialize simplex counts with n vertices
  # The maximum simplex dimension is either n-1 or maxDim
  SimplexCounts = np.zeros( (math.ceil(len(E)*p) + 1 , min(maxDim+1, n)) )
  # Initialize simplex counts to include n initial vertices
  SimplexCounts[0][0] = n

  # Initialize cycle rank and clustering coefficient arrays
  # At time 0: no edges, so cycle rank = 0 - n + n = 0; clustering = 0
  cycleR = np.zeros(math.ceil(len(E)*p) + 1)
  clusterC = np.zeros(math.ceil(len(E)*p) + 1)
  clust_sum = 0.0
  clust_contrib = {i:0.0 for i in range(n)}

  # counter keeps track of first unfilled index in Times, timer keeps track of the step at which a simplex is added
  # The value of timer counts the total number of edges added so far
  counter = n; timer = 0;

  if timing:
    end = time.time()
    print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(2/5) Beginning network evolution",flush=True)
    start = time.time()

  # One by one add each edge e = (u,v) to the graph
  for e in tqdm(E) if timing == True else E:
    # Copy simplex counts from previous step and update timer
    timer += 1; SimplexCounts[timer] = SimplexCounts[timer-1].copy();

    # Add selected edge to graph
    G.add_edge(e[0],e[1]);

    # Compute cycle rank: xi = |E| - |V| + num_connected_components
    # equivalently xi = |E| - |V| + beta_0
    # We dont find the num of components, since at the end we compute it anyways
    cycleR[timer] = G.number_of_edges() - G.number_of_nodes()

    # Compute average clustering coefficient
    clust_sum = ClusterCoeffUpdate(G,[e[0],e[1]],clusterC,clust_contrib,clust_sum,timer)

    # Find all newly added simplices (all must contain e[0] and e[1] by necessity).
    # For each maximal clique containing e, every valid simplex is exactly
    # {e[0], e[1]} union S, where S is any subset of (clique - {e[0], e[1]}).
    # We iterate over subsets of the remainder only, then append e[0] and e[1],
    Simplices = set([tuple(sorted([e[0], e[1]]))])
    Cliques = nx.find_cliques(G, [e[0], e[1]])
    for clique in Cliques:
        remainder = [node for node in clique if node != e[0] and node != e[1]]
        numRemainder = len(remainder)
        # r here is the number of nodes drawn from remainder.
        # r=0 gives the edge {e[0],e[1]} which is already in Simplices.
        # We cap at maxDim-1 since e[0] and e[1] will bring the total to maxDim+1 nodes.
        for r in range(1, min(numRemainder, maxDim - 1) + 1):
            for face in combinations(remainder, r):
                Simplices.add(tuple(sorted(face + (e[0], e[1]))))

    # From set of simplices extract simplex counts, and if the simplex
    # is a tetrahedron or smaller add it to Times
    for simplex in Simplices:
      SimplexCounts[timer][len(simplex)-1] += 1
      if len(simplex) <= 4:
        Times.append((list(simplex), timer));

    # Check if the edge density is greater than p, if so break out of loop
    if timer >= len(E) * p: # timer = num edges added, floor(n*p) = terminal num edges
      break

  if timing:
    end = time.time()
    print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(3/5) Beginning Persistent Homology",flush=True)
    start = time.time()

  # Create filtration of simplicial complexes using Times
  f = d.Filtration(Times)
  # Clear out Times, which is massive
  del(Times); gc.collect()
  # Compute persistent homology of filtration
  m = d.homology_persistence(f)

  if timing:
    end = time.time()
    print("Persistent Homology complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(4/5) Beginning Betti number extraction",flush=True)
    start = time.time()

  # PH doesn't return betti numbers, it returns persistence pairs
  # Here we loop through pairs, any time between the birth
  # and death of the pair corresponds to the existence of a hole
  Betti = np.zeros((4,timer+1))
  one = np.ones(timer+1)
  dgms = d.init_diagrams(m,f)
  for i, dgm in enumerate(dgms):
    for p in dgm:
      Betti[i][int(p.birth):int(min(p.death,timer+1))] += one[int(p.birth):int(min(p.death,timer+1))]

  if timing:
    end = time.time()
    print("Betti number extraction complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(5/5) Beginning Euler Characteristic extraction",flush=True)
    start = time.time()

  # For each time step, compute the euler characteristic as the alternating
  # sum of simplex counts SUM( (-1)^j * num j-simplices )
  Euler = np.zeros(timer+1)
  for i in range(len(SimplexCounts)):
    for j in range(len(SimplexCounts[i])):
      Euler[i] += np.power(-1,j) * SimplexCounts[i][j]

  # Update cycle rank with b0
  cycleR = [cycleR[i] + Betti[0][i] for i in range(len(Betti[0]))]

  if timing:
    end = time.time()
    print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds",flush=True)

  return Betti, SimplexCounts, Euler, cycleR[:timer+1], clusterC[:timer+1]

def PH_BarabasiAlbert(n, k, timing = False, maxDim = np.inf):
  """
  Simulates an evolving Barabasi-Albert (linear preferential
  attachment) graph: starting from a size-(k+1) star graph, each new
  node connects to k existing nodes chosen with probability
  proportional to degree, computing persistent homology of the
  clique complex, cycle rank, and average clustering coefficient as
  it grows to n nodes.

  Parameters
  ----------
  n : int
    Terminal number of nodes.
  k : int
    Number of edges a newly added node forms.
  timing : bool
    Whether to display a progress bar.
  maxDim : int or float
    Maximum simplex dimension to track.

  Returns
  -------
  Betti : ndarray, shape (4, timer+1)
    Betti[k][t] is the k-th Betti number at timestep t (dim 3 is
    truncated, since simplices above dim 3 are not tracked).
  SimplexCounts : list of list of int
    Simplex counts by size at each timestep.
  Euler : ndarray, shape (timer+1,)
    Euler characteristic at each timestep.
  cycleR : list, length timer+1
    CR_1 (= N_1 - N_0 + beta_0) at each timestep.
  clusterC : ndarray, length timer+1
    Average clustering coefficient at each timestep.
  """

  if timing:
    print("(1/5) Initializing graph, variables and data structures",flush=True)
    start = time.time()

  # Initialize BA graph as star graph with k+1 vertices, node 0 connects to nodes 1, 2, ... , k
  G = nx.star_graph(k)
  V = list(range(n))

  # Each time an edge is connected to a node, add a copy of that
  # node to this list. Then randomly sampling from this list
  # is equivalent to preferential attachment.
  repeated_nodes = ([0] * k) + [i for i in range(1,k+1)]

  # Times keeps track of when simplices were added, for persistent homology
  Times = [([i],0) for i in range(k+1)] + [(list(e),0) for e in list(G.edges())]

  # Initialize cycle rank and clustering coefficient arrays
  # At time 0: k edges, so cycle rank = k - (k+1) + 1 = 0;
  cycleR = np.zeros(n - k + 1)
  clusterC = np.zeros(n - k + 1)
  # Star graph has zero clustering
  clust_sum = 0.0
  clust_contrib = {i:0.0 for i in range(k+1)}

  # Intialize simplex counts with k+1 vertices and k edges
  SimplexCounts = [ [k+1, k] ]
  timer = 0

  if timing:
    end = time.time()
    print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(2/5) Beginning network evolution",flush=True)
    start = time.time()

  for v in tqdm(V[k+1:]) if timing else V[k+1:]:
    # Copy simplex counts from previous step and update timer
    SimplexCounts.append(SimplexCounts[-1].copy()); timer += 1

    # Select k targets in graph using preferential attachment
    targets = set()
    while len(targets) < k:
        x = random.choice(repeated_nodes)
        targets.add(x)
    targets = list(targets)
    repeated_nodes.extend(targets + [v] * k)

    # Add node v and update its degree
    G.add_node(v);
    clust_contrib[v] = 0.0

    # Find all of the newly added simplices (which must contain v by necessity)
    # We use a set() for simplices to avoid double adding simplices from maxial cliques
    Simplices = set([tuple([v])]);

    # Add new edges
    for target in targets:
      G.add_edge(target, v);
      Simplices.add(tuple([target,v]))

    # Compute cycle rank: xi = |E| - |V| + num_connected_components
    # equivalently xi = |E| - |V| + beta_0
    # We dont find the num of components, since at the end we compute it anyways
    cycleR[timer] = G.number_of_edges() - G.number_of_nodes()

    # Compute average clustering coefficient
    clust_sum = ClusterCoeffUpdate(G,targets+[v],clusterC,clust_contrib,clust_sum,timer)

    Cliques = nx.find_cliques(G,[v]) # <- Returns maximal cliques containing v
    for clique in Cliques:
      numNodes = len(clique)
      # If a newly added simplex is larger than the previously largest simplex,
      # it can only be one larger so we increase the size of simplex counts by one
      if ( numNodes > len(SimplexCounts[-1]) ) and ( numNodes <= (maxDim + 1) ):
        SimplexCounts[-1].append(0)
      # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
      # all the way down to the 2-subsets (edges).
      # We do sorted() to avoid double creating simplices
      remainder = [node for node in clique if node != v]
      numRemainder = len(remainder)
      for r in range(min(numRemainder, maxDim), 1, -1):
        # combinations(S,r) is from itertools and returns iterator corresponding
        # to all r-subsets of S.
        for face in combinations(remainder, r):
            Simplices.add(tuple(sorted(face + tuple([v]))))

    # From set of simplices extract simplex counts, and if the simplex
    # is a tetrahedron or smaller add it to Times
    for simplex in Simplices:
      SimplexCounts[-1][len(simplex)-1] += 1
      if len(simplex) <= 4:
        Times.append((list(simplex),timer))

  if timing:
    end = time.time()
    print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(3/5) Beginning Persistent Homology",flush=True)
    start = time.time()

  # Create filtration of simplicial complexes using Times
  f = d.Filtration(Times)
  # Clear out Times, which is massive
  del(Times); gc.collect()
  # Compute persistent homology of filtration
  m = d.homology_persistence(f)

  if timing:
    end = time.time()
    print("Persistent Homology complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(4/5) Beginning Betti number extraction",flush=True)
    start = time.time()

  # PH doesn't return betti numbers, it returns persistence pairs
  # Here we loop through pairs, any time between the birth
  # and death of the pair corresponds to the existence of a hole
  Betti = np.zeros((4,timer+1))
  one = np.ones(timer+1)
  dgms = d.init_diagrams(m,f)
  for i, dgm in enumerate(dgms):
    for p in dgm:
      Betti[i][int(p.birth):int(min(p.death,timer+1))] += one[int(p.birth):int(min(p.death,timer+1))]

  if timing:
    end = time.time()
    print("Betti number extraction complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(5/5) Beginning Euler Characteristic extraction",flush=True)
    start = time.time()

  # For each time step, compute the euler characteristic as the alternating
  # sum of simplex counts SUM( (-1)^j * num j-simplices )
  Euler = np.zeros(timer+1)
  for i in range(len(SimplexCounts)):
    for j in range(len(SimplexCounts[i])):
      Euler[i] += np.power(-1,j) * SimplexCounts[i][j]

  # Update cycle rank with b0
  cycleR = [cycleR[i] + Betti[0][i] for i in range(len(Betti[0]))]

  if timing:
    end = time.time()
    print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds",flush=True)

  return Betti, SimplexCounts, Euler, cycleR[:timer+1], clusterC[:timer+1]

def PH_RandomAttachment(n, k, timing = False, maxDim = np.inf):
  """
  Simulates an evolving random-attachment graph: starting from a
  size-(k+1) star graph, each new node connects to k uniformly
  randomly chosen existing nodes, computing persistent homology of
  the clique complex, cycle rank, and average clustering coefficient
  as it grows to n nodes.

  Parameters
  ----------
  n : int
    Terminal number of nodes.
  k : int
    Number of edges a newly added node forms.
  timing : bool
    Whether to display a progress bar.
  maxDim : int or float
    Maximum simplex dimension to track.

  Returns
  -------
  Betti : ndarray, shape (4, timer+1)
    Betti[k][t] is the k-th Betti number at timestep t (dim 3 is
    truncated, since simplices above dim 3 are not tracked).
  SimplexCounts : list of list of int
    Simplex counts by size at each timestep.
  Euler : ndarray, shape (timer+1,)
    Euler characteristic at each timestep.
  cycleR : list, length timer+1
    CR_1 (= N_1 - N_0 + beta_0) at each timestep.
  clusterC : ndarray, length timer+1
    Average clustering coefficient at each timestep.
  """

  if timing:
    print("(1/5) Initializing graph, variables and data structures",flush=True)
    start = time.time()

  # Initialize graph as star graph with k+1 vertices, node 0 connects to nodes 1, 2, ... , k
  G = nx.star_graph(k)
  V = list(range(n))

  # Times keeps track of when simplices are added, for persistent homology
  Times = [([i],0) for i in range(k+1)] + [(list(e),0) for e in list(G.edges())]

  # Initialize cycle rank and clustering coefficient arrays
  # At time 0: k edges, so cycle rank = k - (k+1) + 1 = 0;
  cycleR = np.zeros(n - k + 1)
  clusterC = np.zeros(n - k + 1)
  # Star graph has zero clustering
  clust_sum = 0.0
  clust_contrib = {i:0.0 for i in range(k+1)}

  # Intialize simplex counts with k+1 vertices and k edges
  SimplexCounts = [ [k+1, k] ]
  timer = 0

  if timing:
    end = time.time()
    print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(2/5) Beginning network evolution",flush=True)
    start = time.time()

  for v in tqdm(V[k+1:]) if timing == True else V[k+1:]:
    # Copy simplex counts from previous step and update timer
    SimplexCounts.append(SimplexCounts[-1].copy()); timer += 1

    # Select k targets in graph randomly
    targets = list(np.random.choice(v, size = k, replace = False))
    G.add_node(v);
    clust_contrib[v] = 0.0

    # Find all of the newly added simplices (which must contain v by necessity)
    # We use a set() for simplices to avoid double adding simplices from maxial cliques
    Simplices = set([tuple([v])]);

    # Add new edges and update target degrees
    for target in targets:
      ##D[target] += 1;
      G.add_edge(target, v);
      Simplices.add(tuple([target,v]))

    # Compute cycle rank: xi = |E| - |V| + num_connected_components
    # equivalently xi = |E| - |V| + beta_0
    # We dont find the num of components, since at the end we compute it anyways
    cycleR[timer] = G.number_of_edges() - G.number_of_nodes()

    # Compute average clustering coefficient
    clust_sum = ClusterCoeffUpdate(G,targets+[v],clusterC,clust_contrib,clust_sum,timer)

    Cliques = nx.find_cliques(G,[v]) # <- Returns maximal cliques containing v
    for clique in Cliques:
      numNodes = len(clique)
      # If a newly added simplex is larger than the previously largest simplex,
      # it can only be one larger so be increase the size of simplex counts by one
      if ( numNodes > len(SimplexCounts[-1]) ) and ( numNodes <= (maxDim + 1) ):
        SimplexCounts[-1].append(0)
      # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
      # all the way down to the 2-subsets (edges).
      # We do sorted() to avoid double creating simplices
      remainder = [node for node in clique if node != v]
      numRemainder = len(remainder)
      for r in range(min(numRemainder, maxDim), 1, -1):
        # combinations(S,r) is from itertools and returns iterator corresponding
        # to all r-subsets of S.
        for face in combinations(remainder, r):
            Simplices.add(tuple(sorted(face + tuple([v]))))

    # From set of simplices extract simplex counts, and if the simplex
    # is a tetrahedron or smaller add it to Times
    for simplex in Simplices:
      SimplexCounts[-1][len(simplex)-1] += 1
      if len(simplex) <= 4:
        Times.append((list(simplex),timer))

  if timing:
    end = time.time()
    print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(3/5) Beginning Persistent Homology",flush=True)
    start = time.time()

  # Create filtration of simplicial complexes using Times
  f = d.Filtration(Times)
  # Clear out Times, which is massive
  del(Times); gc.collect()
  # Compute persistent homology of filtration
  m = d.homology_persistence(f)

  if timing:
    end = time.time()
    print("Persistent Homology complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(4/5) Beginning Betti number extraction",flush=True)
    start = time.time()

  # PH doesn't return betti numbers, it returns persistence pairs
  # Here we loop through pairs, any time between the birth
  # and death of the pair corresponds to the existence of a hole
  Betti = np.zeros((4,timer+1))
  one = np.ones(timer+1)
  dgms = d.init_diagrams(m,f)
  for i, dgm in enumerate(dgms):
    for p in dgm:
      Betti[i][int(p.birth):int(min(p.death,timer+1))] += one[int(p.birth):int(min(p.death,timer+1))]

  if timing:
    end = time.time()
    print("Betti number extraction complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(5/5) Beginning Euler Characteristic extraction",flush=True)
    start = time.time()

  # For each time step, compute the euler characteristic as the alternating
  # sum of simplex counts SUM( (-1)^j * num j-simplices )
  Euler = np.zeros(timer+1)
  for i in range(len(SimplexCounts)):
    for j in range(len(SimplexCounts[i])):
      Euler[i] += np.power(-1,j) * SimplexCounts[i][j]

  # Update cycle rank with b0
  cycleR = [cycleR[i] + Betti[0][i] for i in range(len(Betti[0]))]

  if timing:
    end = time.time()
    print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds",flush=True)

  return Betti, SimplexCounts, Euler, cycleR[:timer+1], clusterC[:timer+1]

def ZZPH_WattsStrogatz(n, m, timing = False, maxDim = np.inf):
  """
  Simulates an evolving Watts-Strogatz graph: starting from a ring
  lattice where each node connects to its m nearest neighbors, the
  original lattice edges are rewired one at a time (in random order,
  each moved to a new uniformly random endpoint), computing zigzag
  persistent homology of the clique complex, average clustering
  coefficient, and average shortest path length over the resulting
  add/remove filtration.

  Parameters
  ----------
  n : int
    Number of nodes.
  m : int
    Degree of each node in the initial ring lattice (even).
  timing : bool
    Whether to display a progress bar.
  maxDim : int or float
    Maximum simplex dimension to track.

  Returns
  -------
  Betti : ndarray, shape (4, timer+1)
    Betti[k][t] is the k-th Betti number at timestep t (dim 3 is
    truncated, since simplices above dim 3 are not tracked).
  SimplexCounts : list of list of int
    Simplex counts by size at each timestep.
  Euler : ndarray, shape (timer+1,)
    Euler characteristic at each timestep.
  clusterC : ndarray, length timer+1
    Average clustering coefficient at each timestep.
  pathL : ndarray, length timer+1
    Average shortest path length at each timestep (inf if the graph
    is disconnected).
  """

  if timing:
    print("(1/5) Initializing graph, variables and data structures",flush=True)
    start = time.time()

  # Intialize ring lattice and list of original edges of ring lattice
  # which will be iterated over and rewired
  G = nx.watts_strogatz_graph(n,m,0)
  E = list(G.edges())

  # Initialize array for simplex counts at each step. In the initial ring
  # lattice there are simplices up to dimension m/2
  SimplexCounts = [ [0] * (int(m/2) + 1) ]

  # Initialize dictionary which keeps track of when simplices
  # are added and removed
  Times = {tuple(clique) : [0] for clique in nx.enumerate_all_cliques(G)}
  for simplex in Times:
    SimplexCounts[0][len(simplex)-1] += 1

  # We uniformly select edges by shuffling E and iterating through it
  # timer = total number of edges rewired by end of current step
  random.shuffle(E); timer = 0

  clusterC = np.zeros(len(E)+1)
  clust_contrib = nx.clustering(G)
  clust_sum = sum(clust_contrib.values())
  clusterC[0] = clust_sum / n
  pathL = np.zeros(len(E)+1)
  pathL[0] = nx.average_shortest_path_length(G)

  if timing:
    end = time.time()
    print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(2/5) Beginning network evolution",flush=True)
    start = time.time()

  # Iterate until each of the original lattice edges has been rewired
  for e in tqdm(E) if timing == True else E:
    # Copy simplex counts from previous step and update timer
    SimplexCounts.append(SimplexCounts[-1].copy()); timer += 1;

    # Select which end of the edge to keep (source) and which to rewire away from (target)
    choice = random.choice([0,1])
    source = e[choice]; target = e[(choice + 1) % 2]
    # Select new end point uniformly from vertices not connected to the source
    while True:
      newTarget = random.randrange(n)
      if (newTarget not in G.neighbors(source)) and (newTarget != source):
        break

    # Removing the edge (source, target) removes simplices, so we find these
    # simplices and remove them. We use a set() for simplices to avoid double
    # adding simplices from maxial cliques, and initialize Simplices to
    # contain (source,target) to reduce need for computation
    Simplices = set([tuple(sorted([source,target]))]); Cliques = nx.find_cliques(G,sorted([source, target])) # <- Returns maximal cliques containing e
    for clique in Cliques:
      # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
      # all the way down to the 3-subsets (triangles). We don't do edges or nodes
      # since we never add/remove nodes, and the only edge we care about has
      # already been added. This reduces computation.
      # We do sorted() to avoid double creating simplices
      remainder = [node for node in clique if node != source and node != target]
      numRemainder = len(remainder)
      for r in range(1, min(numRemainder, maxDim - 1) + 1):
        for face in combinations(remainder, r):
          Simplices.add(tuple(sorted(face + (source, target))))

    # From set of simplices extract simplex counts, and if the simplex
    # is a tetrahedron or smaller add it to Times
    for simplex in Simplices:
      SimplexCounts[-1][len(simplex)-1] -= 1
      if len(simplex) <= 4:
        # Since we are removing the simplex, it must already exist in Times dict
        Times[simplex].append(timer)

    # Remove edge e from G
    G.remove_edge(e[0],e[1]);

    # Add edge (source,newTarget) to G
    G.add_edge(source,newTarget);

    # Compute average clustering coefficient
    clust_sum = ClusterCoeffUpdate(G,[source,newTarget],clusterC,clust_contrib,clust_sum,timer)
    # Compute average shortest path length
    if nx.is_connected(G):
        pathL[timer] = nx.average_shortest_path_length(G)
    else:
        pathL[timer] = np.inf

    # Find all of the newly added simplices (which must contain source and newTarget by necessity)
    # We use a set() for simplices to avoid double adding simplices from maxial cliques,
    # and initialize Simplices to contain (source,newTarget) to reduce need for computation
    Simplices = set([tuple(sorted([source,newTarget]))]); Cliques = nx.find_cliques(G,sorted([source,newTarget])) # <- Returns maximal cliques containing source and newTarget
    for clique in Cliques:
      numNodes = len(clique)
      # If a newly added simplex is larger than the previously largest simplex,
      # it can only be one larger so be increase the size of simplex counts by one
      if ( numNodes > len(SimplexCounts[-1]) ) and ( numNodes <= (maxDim + 1) ):
        SimplexCounts[-1].append(0)
      # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
      # all the way down to the 3-subsets (triangles). We don't do edges or nodes
      # since we never add/remove nodes, and the only edge we care about has
      # already been added. This reduces computation.
      remainder = [node for node in clique if node != source and node != newTarget]
      numRemainder = len(remainder)
      for r in range(1, min(numRemainder, maxDim - 1) + 1):
        for face in combinations(remainder, r):
          Simplices.add(tuple(sorted(face + (source, newTarget))))


    # From set of simplices extract simplex counts, and if the simplex
    # is a tetrahedron or smaller add it to Times
    for simplex in Simplices:
      dim = len(simplex) - 1
      SimplexCounts[-1][dim] += 1
      if dim <= 3:
        # Since we are adding simplices to complex, we don't know if they
        # previously exists and were then removed, i.e. we need to check if
        # they already exist in Times dict
        if simplex in Times:
          Times[simplex].append(timer)
        else:
          Times[simplex] = [timer]

  if timing:
    end = time.time()
    print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(3/5) Beginning Zigzag Persistent Homology",flush=True)
    start = time.time()

  # Extract list of every simplex added/removed, and list of times they were
  # added/removed, for input into zigzag persistence.
  simplices = [list(key) for key in Times]; times = [Times[key] for key in Times]

  # Clear out Times, which is massive
  del(Times); del(G); del(E); gc.collect()

  # Construct filtration and compute homology
  f = d.Filtration(simplices)
  zz, dgms, cells = d.zigzag_homology_persistence(f, times)

  # Clear out remaining lists which are massive
  del(simplices); del(times); gc.collect()

  if timing:
    end = time.time()
    print("Zigzag Persistent Homology complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(4/5) Beginning Betti number extraction",flush=True)
    start = time.time()

  # PH doesn't return betti numbers, it returns persistence pairs
  # Here we loop through pairs, any time between the birth
  # and death of the pair corresponds to the existence of a hole
  Betti = np.zeros((4,timer+1))
  one = np.ones(timer+1)
  for i, dgm in enumerate(dgms):
    for p in dgm:
      Betti[i][int(p.birth):int(min(p.death,timer+1))] += one[int(p.birth):int(min(p.death,timer+1))]

  if timing:
    end = time.time()
    print("Betti number extraction complete, time taken : "+str(end - start)+" seconds")
    print("(5/5) Beginning Euler Characteristic extraction")
    start = time.time()

  # For each time step, compute the euler characteristic as the alternating
  # sum of simplex counts SUM( (-1)^j * num j-simplices )
  Euler = np.zeros(timer+1)
  for i in range(len(SimplexCounts)):
    for j in range(len(SimplexCounts[i])):
      Euler[i] += np.power(-1,j) * SimplexCounts[i][j]

  if timing:
    end = time.time()
    print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds")

  return Betti, SimplexCounts, Euler, clusterC[:timer+1], pathL[:timer+1]

def PH_Nonlinear_BarabasiAlbert(n, k, alpha = 1.0, timing = False, maxDim = np.inf):
  """
  Simulates an evolving nonlinear-preferential-attachment graph: like
  PH_BarabasiAlbert, but a node's attachment weight is its degree
  raised to the power alpha (alpha=1 recovers linear preferential
  attachment; alpha=0 recovers uniform/random attachment). Computes
  persistent homology of the clique complex, cycle rank, and average
  clustering coefficient as it grows to n nodes.

  Parameters
  ----------
  n : int
    Terminal number of nodes.
  k : int
    Number of edges a newly added node forms.
  alpha : float
    Nonlinear preferential-attachment exponent applied to node degree.
  timing : bool
    Whether to display a progress bar.
  maxDim : int or float
    Maximum simplex dimension to track.

  Returns
  -------
  Betti : ndarray, shape (4, timer+1)
    Betti[k][t] is the k-th Betti number at timestep t (dim 3 is
    truncated, since simplices above dim 3 are not tracked).
  SimplexCounts : list of list of int
    Simplex counts by size at each timestep.
  Euler : ndarray, shape (timer+1,)
    Euler characteristic at each timestep.
  D : collections.Counter
    Degree frequency distribution: maps each observed degree to the
    number of nodes with that degree.
  cycleR : list, length timer+1
    CR_1 (= N_1 - N_0 + beta_0) at each timestep.
  clusterC : ndarray, length timer+1
    Average clustering coefficient at each timestep.
  """
  if timing:
    print("(1/5) Initializing graph, variables and data structures",flush=True)
    start = time.time()

  # Initialize BA graph as star graph with k+1 vertices, node 0 connects to nodes 1, 2, ... , k
  G = nx.star_graph(k)
  V = list(range(n))

  # Keep track of vertex degrees for degree distribution
  D = [k] + [1] * k

  # Initialize cycle rank and clustering coefficient arrays
  # At time 0: k edges, so cycle rank = k - (k+1) + 1 = 0;
  cycleR = np.zeros(n - k + 1)
  clusterC = np.zeros(n - k + 1)
  # Star graph has zero clustering
  clust_sum = 0.0
  clust_contrib = {i:0.0 for i in range(k+1)}

  # Initialize efficient weighted sampler
  initial_weights = [k] + [1] * k
  sampler = WeightedSampler(initial_weights)
  # We have k+1 entries in D, so we extend to n by adding n - k - 1 entries
  D.extend([0] * (n - k - 1))

  # Times keeps track of when simplices were added, for persistent homology
  Times = [([i],0) for i in range(k+1)] + [(list(e),0) for e in list(G.edges())]

  # Intialize simplex counts with k+1 vertices and k edges
  SimplexCounts = [ [k+1, k] ]
  timer = 0

  if timing:
    end = time.time()
    print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(2/5) Beginning network evolution",flush=True)
    start = time.time()

  for v in tqdm(V[k+1:]) if timing else V[k+1:]:
    # Copy simplex counts from previous step and update timer
    SimplexCounts.append(SimplexCounts[-1].copy()); timer += 1

    # Select k targets in graph using preferential attachment
    targets = set()

    while len(targets) < k:
      x = sampler.sample(exclude=targets)
      if x is not None:
        targets.add(x)
      else:
        break

    targets = list(targets)

    # Add node v and update its degree
    G.add_node(v);
    clust_contrib[v] = 0.0
    D[v] = k
    new_weight = D[v] ** alpha
    sampler.update_weight(v, new_weight)

    # Find all of the newly added simplices (which must contain v by necessity)
    # We use a set() for simplices to avoid double adding simplices from maxial cliques
    Simplices = set([tuple([v])]);

    # Add new edges
    for target in targets:
      D[target] += 1
      new_degree = D[target]
      new_weight = new_degree ** alpha
      sampler.update_weight(target, new_weight)
      G.add_edge(target, v);
      Simplices.add(tuple([target,v]))

    # Compute cycle rank: xi = |E| - |V| + num_connected_components
    # equivalently xi = |E| - |V| + beta_0
    # We dont find the num of components, since at the end we compute it anyways
    cycleR[timer] = G.number_of_edges() - G.number_of_nodes() #+ nx.number_connected_components(G)

    # Compute average clustering coefficient
    clust_sum = ClusterCoeffUpdate(G,targets+[v],clusterC,clust_contrib,clust_sum,timer)

    Cliques = nx.find_cliques(G,[v]) # <- Returns maximal cliques containing v
    for clique in Cliques:
      numNodes = len(clique)
      # If a newly added simplex is larger than the previously largest simplex,
      # it can only be one larger so we increase the size of simplex counts by one
      if ( numNodes > len(SimplexCounts[-1]) ) and ( numNodes <= (maxDim + 1) ):
        SimplexCounts[-1].append(0)
      # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
      # all the way down to the 2-subsets (edges).
      # We do sorted() to avoid double creating simplices
      clique = sorted(clique)
      remainder = [node for node in clique if node != v]
      numRemainder = len(remainder)
      for r in range(1, min(numRemainder, maxDim) + 1):
        for face in combinations(remainder, r):
            Simplices.add(tuple(sorted(face + tuple([v]))))

    # From set of simplices extract simplex counts, and if the simplex
    # is a tetrahedron or smaller add it to Times
    for simplex in Simplices:
      SimplexCounts[-1][len(simplex)-1] += 1
      if len(simplex) <= 4:
        Times.append((list(simplex),timer))

  if timing:
    end = time.time()
    print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(3/5) Beginning Persistent Homology",flush=True)
    start = time.time()

  # Create filtration of simplicial complexes using Times
  f = d.Filtration(Times)
  # Clear out Times, which is massive
  del(Times); gc.collect()
  # Compute persistent homology of filtration
  m = d.homology_persistence(f)

  if timing:
    end = time.time()
    print("Persistent Homology complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(4/5) Beginning Betti number extraction",flush=True)
    start = time.time()

  # PH doesn't return betti numbers, it returns persistence pairs
  # Here we loop through pairs, any time between the birth
  # and death of the pair corresponds to the existence of a hole
  Betti = np.zeros((4,timer+1))
  one = np.ones(timer+1)
  dgms = d.init_diagrams(m,f)
  for i, dgm in enumerate(dgms):
    for p in dgm:
      Betti[i][int(p.birth):int(min(p.death,timer+1))] += one[int(p.birth):int(min(p.death,timer+1))]

  if timing:
    end = time.time()
    print("Betti number extraction complete, time taken : "+str(end - start)+" seconds",flush=True)
    print("(5/5) Beginning Euler Characteristic extraction",flush=True)
    start = time.time()

  # For each time step, compute the euler characteristic as the alternating
  # sum of simplex counts SUM( (-1)^j * num j-simplices )
  Euler = np.zeros(timer+1)
  for i in range(len(SimplexCounts)):
    for j in range(len(SimplexCounts[i])):
      Euler[i] += np.power(-1,j) * SimplexCounts[i][j]

  # Update cycle rank with b0
  cycleR = [cycleR[i] + Betti[0][i] for i in range(len(Betti[0]))]

  if timing:
    end = time.time()
    print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds",flush=True)

  D = collections.Counter(D)
  return Betti, SimplexCounts, Euler, D, cycleR[:timer+1], clusterC[:timer+1]

# Example usage with different alpha values:
# alpha = 1.0  # Linear preferential attachment (original behavior)
# alpha > 1.0  # Super-linear (rich get richer effect amplified)
# alpha < 1.0  # Sub-linear (more egalitarian attachment)
# alpha = 0.0  # Uniform random attachment

# ============================================================================
# Parallel calls
# ============================================================================

def ParallelCall_ER(params):
  """
  Worker function for parallel (joblib) sweeps of the Erdos-Renyi
  graph model: runs PH_ErdosRenyi for one parameter combination and
  pickles the result. Skips the run if its output file already exists.

  Parameters
  ----------
  params : tuple
    (n, p, iteration): node count, terminal edge density, run index.

  Returns
  -------
  int
    0 in all cases. Results ([Betti, SimplexCounts, Euler, cycleR,
    clusterC]) are written to a filename derived from params rather
    than returned directly.
  """
  n = params[0]; p = params[1]; iteration = params[2]
  filename = 'ErdosRenyi/ER_'+str(n)+'_'+str(p).replace('.','_')+'_'+str(iteration)+'.pkl'

  if os.path.isfile(filename):
      return 0

  Betti, SimplexCounts, Euler, cycleR, clusterC = PH_ErdosRenyi(n, p, timing = False)
  Data = [Betti, SimplexCounts, Euler, cycleR, clusterC]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(Data, f);
  return 0

def ParallelCall_BA(params):
  """
  Worker function for parallel (joblib) sweeps of the Barabasi-Albert
  graph model: runs PH_BarabasiAlbert for one parameter combination
  and pickles the result. Skips the run if its output file already
  exists.

  Parameters
  ----------
  params : tuple
    (n, k, iteration): terminal node count, attachment count, run index.

  Returns
  -------
  int
    0 in all cases. Results ([Betti, SimplexCounts, Euler, cycleR,
    clusterC]) are written to a filename derived from params rather
    than returned directly.
  """
  n = params[0]; k = params[1]; iteration = params[2]
  filename = 'BarabasiAlbert/BA_'+str(n)+'_'+str(k)+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
      return 0
  Betti, SimplexCounts, Euler, cycleR, clusterC = PH_BarabasiAlbert(n,k,False)
  Data = [Betti, SimplexCounts, Euler, cycleR, clusterC]
  output = open(filename,'wb')
  pickle.dump(Data, output); output.close()
  return 0

def ParallelCall_RA(params):
  """
  Worker function for parallel (joblib) sweeps of the random-attachment
  graph model: runs PH_RandomAttachment for one parameter combination
  and pickles the result. Skips the run if its output file already
  exists.

  Parameters
  ----------
  params : tuple
    (n, k, iteration): terminal node count, attachment count, run index.

  Returns
  -------
  int
    0 in all cases. Results ([Betti, SimplexCounts, Euler, cycleR,
    clusterC]) are written to a filename derived from params rather
    than returned directly.
  """
  n = params[0]; k = params[1]; iteration = params[2]
  filename = 'RandomAttachment/RA_'+str(n)+'_'+str(k)+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
      return 0
  Betti, SimplexCounts, Euler, cycleR, clusterC = PH_RandomAttachment(n,k,False)
  Data = [Betti, SimplexCounts, Euler, cycleR, clusterC]
  output = open(filename,'wb')
  pickle.dump(Data, output); output.close()
  return 0

def ParallelCall_WS(params):
  """
  Worker function for parallel (joblib) sweeps of the Watts-Strogatz
  graph model: runs ZZPH_WattsStrogatz for one parameter combination
  and pickles the result. Skips the run if its output file already
  exists.

  Parameters
  ----------
  params : tuple
    (n, k, iteration): node count, ring-lattice degree, run index.

  Returns
  -------
  int
    0 in all cases. Results ([Betti, SimplexCounts, Euler, clusterC,
    pathL]) are written to a filename derived from params rather than
    returned directly.
  """
  n = params[0]; k = params[1]; iteration = params[2]
  filename = 'WattsStrogatz/WS_'+str(n)+'_'+str(k)+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
    return 0
  Betti, SimplexCounts, Euler, clusterC, pathL = ZZPH_WattsStrogatz(n,k)
  Data = [Betti, SimplexCounts, Euler, clusterC, pathL]
  output = open(filename,'wb')
  pickle.dump(Data, output); output.close()
  return 0

def ParallelCall_NLPA(params):
  """
  Worker function for parallel (joblib) sweeps of the nonlinear
  preferential-attachment graph model: runs
  PH_Nonlinear_BarabasiAlbert for one parameter combination and
  pickles the result. Skips the run if its output file already exists.

  Parameters
  ----------
  params : tuple
    (n, k, alpha, iteration): terminal node count, attachment count,
    nonlinear preferential-attachment exponent, and run index.

  Returns
  -------
  int
    0 in all cases. Results ([Betti, SimplexCounts, Euler, D,
    cycleR, clusterC]) are written to a filename derived from params
    rather than returned directly.
  """
  n = params[0]; k = params[1]; alpha = params[2]; iteration = params[3]
  filename = 'NLPA/NLPA_'+str(n)+'_'+str(k)+'_'+str(alpha).replace('.','_')+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
      return 0
  Betti, SimplexCounts, Euler, D, cycleR, clusterC = PH_Nonlinear_BarabasiAlbert(n,k,alpha,False)
  Data = [Betti, SimplexCounts, Euler, D, cycleR, clusterC]
  output = open(filename,'wb')
  pickle.dump(Data, output); output.close()
  return 0

# ============================================================================
# Loading/Plotting
# ============================================================================

def load_single_er(N, p, iteration, base_path):
    """
    Loads one gzip-pickled Erdos-Renyi result file (see
    ParallelCall_ER), tolerating a missing or corrupted file.

    Parameters
    ----------
    N : int
      Node count used to locate the file.
    p : float
      Terminal edge density used to locate the file.
    iteration : int
      Run index used to locate the file.
    base_path : str
      Directory the file is located in.

    Returns
    -------
    N : int
      Echoed input node count.
    p : float
      Echoed input edge density.
    iteration : int
      Echoed input run index.
    entry : list or None
      The loaded [Betti, SimplexCounts, Euler, cycleR, clusterC]
      result, or None on failure.
    status : str
      'ok', 'missing', or 'error: <exception message>'.
    """
    filename = os.path.join(base_path,f'ER_{N}_{str(p).replace(".","_")}_{iteration}.pkl')
    if not os.path.isfile(filename):
        return N, p, iteration, None, 'missing'
    try:
        with gzip.open(filename, 'rb') as f:
            entry = pickle.load(f)
        return N, p, iteration, entry, 'ok'
    except Exception as e:
        return N, p, iteration, None, f'error: {e}'

def load_single_ws(N, k, iteration, base_path):
    """
    Loads one pickled (not gzip-compressed, matching how
    ParallelCall_WS saves it) Watts-Strogatz result file, tolerating
    a missing or corrupted file.

    Parameters
    ----------
    N : int
      Node count used to locate the file.
    k : int
      Ring-lattice degree used to locate the file.
    iteration : int
      Run index used to locate the file.
    base_path : str
      Directory the file is located in.

    Returns
    -------
    N : int
      Echoed input node count.
    k : int
      Echoed input ring-lattice degree.
    iteration : int
      Echoed input run index.
    entry : list or None
      The loaded [Betti, SimplexCounts, Euler, clusterC, pathL]
      result, or None on failure.
    status : str
      'ok', 'missing', or 'error: <exception message>'.
    """
    filename = os.path.join(
        base_path,
        f'WS_{N}_{k}_{iteration}.pkl'
    )
    if not os.path.isfile(filename):
        return N, k, iteration, None, 'missing'
    try:
        with open(filename, 'rb') as f:
            entry = pickle.load(f)
        return N, k, iteration, entry, 'ok'
    except Exception as e:
        return N, k, iteration, None, f'error: {e}'

def load_single_ba(N, k, iteration, base_path):
    """
    Loads one pickled (not gzip-compressed, matching how
    ParallelCall_BA saves it) Barabasi-Albert result file, tolerating
    a missing or corrupted file. Note: unlike load_single_er/ws, N is
    not included in the return value.

    Parameters
    ----------
    N : int
      Node count used to locate the file (not returned).
    k : int
      Attachment count used to locate the file.
    iteration : int
      Run index used to locate the file.
    base_path : str
      Directory the file is located in.

    Returns
    -------
    k : int
      Echoed input attachment count.
    iteration : int
      Echoed input run index.
    entry : list or None
      The loaded [Betti, SimplexCounts, Euler, cycleR, clusterC]
      result, or None on failure.
    status : str
      'ok', 'missing', or 'error: <exception message>'.
    """
    filename = os.path.join(base_path,f'BA_{N}_{k}_{iteration}.pkl')
    if not os.path.isfile(filename):
        return k, iteration, None, 'missing'
    try:
        with open(filename, 'rb') as f:
            entry = pickle.load(f)
        return k, iteration, entry, 'ok'
    except Exception as e:
        return k, iteration, None, f'error: {e}'

def load_single_ra(N, k, iteration, base_path):
    """
    Loads one pickled (not gzip-compressed, matching how
    ParallelCall_RA saves it) random-attachment result file,
    tolerating a missing or corrupted file. Note: unlike
    load_single_er/ws, N is not included in the return value.

    Parameters
    ----------
    N : int
      Node count used to locate the file (not returned).
    k : int
      Attachment count used to locate the file.
    iteration : int
      Run index used to locate the file.
    base_path : str
      Directory the file is located in.

    Returns
    -------
    k : int
      Echoed input attachment count.
    iteration : int
      Echoed input run index.
    entry : list or None
      The loaded [Betti, SimplexCounts, Euler, cycleR, clusterC]
      result, or None on failure.
    status : str
      'ok', 'missing', or 'error: <exception message>'.
    """
    filename = os.path.join(base_path,f'RA_{N}_{k}_{iteration}.pkl')
    if not os.path.isfile(filename):
        return k, iteration, None, 'missing'
    try:
        with open(filename, 'rb') as f:
            entry = pickle.load(f)
        return k, iteration, entry, 'ok'
    except Exception as e:
        return k, iteration, None, f'error: {e}'

def load_single_nlpa(N, k, alpha, iteration, base_path):
    """
    Loads one pickled (not gzip-compressed, matching how
    ParallelCall_NLPA saves it) nonlinear-preferential-attachment
    result file, tolerating a missing or corrupted file. Note: unlike
    load_single_er/ws, N is not included in the return value.

    Parameters
    ----------
    N : int
      Node count used to locate the file (not returned).
    k : int
      Attachment count used to locate the file.
    alpha : float
      Nonlinear preferential-attachment exponent used to locate the file.
    iteration : int
      Run index used to locate the file.
    base_path : str
      Directory the file is located in.

    Returns
    -------
    k : int
      Echoed input attachment count.
    alpha : float
      Echoed input exponent.
    iteration : int
      Echoed input run index.
    entry : list or None
      The loaded [Betti, SimplexCounts, Euler, D, cycleR, clusterC]
      result, or None on failure.
    status : str
      'ok', 'missing', or 'error: <exception message>'.
    """
    filename = os.path.join(
        base_path,
        f'NLPA_{N}_{k}_{str(alpha).replace(".","_")}_{iteration}.pkl'
    )
    if not os.path.isfile(filename):
        return k, alpha, iteration, None, 'missing'
    try:
        with open(filename, 'rb') as f:
            entry = pickle.load(f)
        return k, alpha, iteration, entry, 'ok'
    except Exception as e:
        return k, alpha, iteration, None, f'error: {e}'

def sc_col(entry, dim):
    """
    Extracts one dimension's simplex-count time series from a loaded
    ER/BA/RA/NLPA result entry's SimplexCounts field, which per
    CLAUDE.md is a list of lists (possibly ragged across timesteps),
    NOT a 2D numpy array.

    Parameters
    ----------
    entry : list
      A loaded result, indexed as entry[1] = SimplexCounts.
    dim : int
      Column index (simplex size - 1) to extract.

    Returns
    -------
    ndarray
      1-D array, one entry per timestep, with 0 where a row is too
      short to have that column.
    """
    sc = entry[1]
    return np.array([
        row[dim] if len(row) > dim else 0
        for row in sc
    ], dtype=float)

def cR1(e):
    """
    Extracts the stored CR_1 (= N_1 - N_0 + beta_0) time series from a
    loaded ER/BA/RA result entry.

    Parameters
    ----------
    e : list
      A loaded result, indexed as e[3] = cycleR.

    Returns
    -------
    ndarray
      1-D array of CR_1 at each timestep.
    """
    return np.array(e[3], dtype=float)

def cR2(e):
    """
    Computes the CR_2 time series from a loaded ER/BA/RA result entry:
    CR_2 = N_2 - CR_1 + beta_1.

    Parameters
    ----------
    e : list
      A loaded result, indexed as e[0] = Betti, e[1] = SimplexCounts.

    Returns
    -------
    ndarray
      1-D array of CR_2 at each timestep.
    """
    return sc_col(e, 2) - cR1(e) + e[0][1].astype(float)

def cR1_nlpa(e):
    """
    Extracts the stored CR_1 (= N_1 - N_0 + beta_0) time series from a
    loaded NLPA result entry (whose fields are offset by one relative
    to ER/BA/RA entries, due to the extra degree-distribution field D).

    Parameters
    ----------
    e : list
      A loaded NLPA result, indexed as e[4] = cycleR.

    Returns
    -------
    ndarray
      1-D array of CR_1 at each timestep.
    """
    return np.array(e[4], dtype=float)

def cR2_nlpa(e):
    """
    Computes the CR_2 time series from a loaded NLPA result entry:
    CR_2 = N_2 - CR_1 + beta_1.

    Parameters
    ----------
    e : list
      A loaded NLPA result, indexed as e[0] = Betti, e[1] = SimplexCounts.

    Returns
    -------
    ndarray
      1-D array of CR_2 at each timestep.
    """
    return sc_col(e, 2) - cR1_nlpa(e) + e[0][1].astype(float)

def sc_col_ws(sc_list, dim):
    """
    Extracts one dimension's simplex-count time series from a
    Watts-Strogatz result's SimplexCounts field (a list of lists,
    per CLAUDE.md, possibly containing None rows).

    Parameters
    ----------
    sc_list : list of list of float, or None
      SimplexCounts time series; sc_list[t] is the row of simplex
      counts by size at timestep t.
    dim : int
      Column index (simplex size - 1) to extract.

    Returns
    -------
    ndarray
      1-D array, one entry per timestep, with 0 where a row is
      missing or too short to have that column; a length-1 zero
      array if sc_list itself is None.
    """
    if sc_list is None:
        return np.zeros(1)
    return np.array([
        row[dim] if (row is not None and len(row) > dim) else 0
        for row in sc_list
    ], dtype=float)

def cR1_ws(e):
    """
    Computes the CR_1 (= N_1 - N_0 + beta_0) time series for a loaded
    Watts-Strogatz result entry (which, unlike the ER/BA/RA/NLPA
    models, does not store cycleR directly).

    Parameters
    ----------
    e : list
      A loaded WS result, indexed as e[0] = Betti, e[1] = SimplexCounts.

    Returns
    -------
    ndarray
      1-D array of CR_1 at each timestep.
    """
    return (sc_col_ws(e[1], 1)
            - sc_col_ws(e[1], 0)
            + e[0][0].astype(float))

def cR2_ws(e):
    """
    Computes the CR_2 time series for a loaded Watts-Strogatz result
    entry: CR_2 = N_2 - CR_1 + beta_1.

    Parameters
    ----------
    e : list
      A loaded WS result, indexed as e[0] = Betti, e[1] = SimplexCounts.

    Returns
    -------
    ndarray
      1-D array of CR_2 at each timestep.
    """
    return sc_col_ws(e[1], 2) - cR1_ws(e) + e[0][1].astype(float)

def get_mean_traj_er(data, n, p, iterations, field_fn):
    """
    Computes the mean, across iterations, of a field's trajectory for
    one Erdos-Renyi n/p combination, interpolated onto a common
    x grid spanning [0, p].

    Parameters
    ----------
    data : dict
      Nested results dict indexed as data[n][p].
    n : int
      Node count to select.
    p : float
      Terminal edge density to select.
    iterations : iterable
      Run indices to average over.
    field_fn : callable
      Function (entry) -> 1-D trajectory, e.g. cR1 or cR2.

    Returns
    -------
    x_common : ndarray or None
      Common edge-density grid spanning [0, p], or None if no
      iteration produced a valid trajectory.
    mean : ndarray or None
      Mean trajectory interpolated onto x_common, or None likewise.
    """
    trajs = []
    for iteration in iterations:
        entry = data[n][p].get(iteration, None)
        if entry is None: continue
        try:
            traj = field_fn(entry)
            if traj is not None:
                trajs.append(np.array(traj, dtype=float))
        except Exception: continue
    if not trajs:
        return None, None
    max_len  = max(len(t) for t in trajs)
    x_common = np.linspace(0, p, max_len)
    interp   = np.array([
        np.interp(x_common, np.linspace(0, p, len(t)), t)
        for t in trajs
    ])
    return x_common, interp.mean(axis=0)

def add_discrete_cbar_er(fig, ax_target, n_cmap, n_norm, n_n, N, shrink=0.6):
    """
    Adds a discrete colorbar labeled by node count N to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
      Figure to attach the colorbar to.
    ax_target : Axes or array of Axes
      Axes the colorbar is placed relative to.
    n_cmap : Colormap
      Colormap used for the N values.
    n_norm : Normalize
      Normalization mapping N values to [0, 1].
    n_n : int
      Number of discrete N values (tick count).
    N : list
      N values, in tick order, used as tick labels.
    shrink : float
      Colorbar shrink factor.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
      The created colorbar.
    """
    sm = cm.ScalarMappable(cmap=n_cmap, norm=n_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_target, shrink=shrink, pad=0.02,
                         location='right', ticks=np.arange(n_n))
    cbar.set_ticklabels([str(n) for n in N])
    cbar.set_label(r'$N$ (number of nodes)', fontsize=12)
    return cbar

def draw_er_figure(data_er, n_colors_all, NP_sorted, fields_er_pub, iterations, n_cmap_all, n_norm_all, n_k, N_unique, panel_layout, suptitle, base_path, save_prefix, shrink=0.5):
    """
    Plots a grid of panels, each showing mean field trajectories vs.
    edge density for the Erdos-Renyi model, one line per (N, p)
    combination (colored by N), and saves the figure.

    Parameters
    ----------
    data_er : dict
      Nested results dict indexed as data_er[n][p] (see get_mean_traj_er).
    n_colors_all : dict
      Maps each N value to a plot color.
    NP_sorted : list of tuple
      (n, p) combinations to plot, one line per pair.
    fields_er_pub : dict
      Maps each panel label to a field_fn (see get_mean_traj_er).
    iterations : iterable
      Run indices to average over.
    n_cmap_all : Colormap
      Colormap for the N colorbar.
    n_norm_all : Normalize
      Normalization for the N colorbar.
    n_k : int
      Number of discrete N values (colorbar tick count).
    N_unique : list
      N values, in tick order, for the colorbar.
    panel_layout : list of tuple of str
      Rows of (left_label, right_label) panel labels, keys into
      fields_er_pub.
    suptitle : str
      Figure title.
    base_path : str
      Directory to save the figure files to.
    save_prefix : str
      Filename prefix (without extension) for the saved figure.
    shrink : float
      Colorbar shrink factor.

    Returns
    -------
    None
      Displays the figure and saves it as PDF and PNG under base_path.
    """
    n_rows = len(panel_layout)
    fig, axes = plt.subplots(n_rows, 2,
                              figsize=(10, 3.5 * n_rows),
                              constrained_layout=True)

    for row_idx, (left_label, right_label) in enumerate(panel_layout):
        is_bottom = (row_idx == n_rows - 1)
        for col_idx, label in enumerate([left_label, right_label]):
            ax       = axes[row_idx][col_idx]
            field_fn = fields_er_pub[label]

            for n, p in NP_sorted:
                x, mean = get_mean_traj_er(data_er, n, p, iterations, field_fn)
                if x is None: continue
                ax.plot(x, mean,
                        color=n_colors_all[n],
                        lw=1.2, alpha=0.9)

            ax.set_ylabel(label)
            ax.set_xlim(0, 1)
            ax.set_xlabel(r'Edge density $p$')

    add_discrete_cbar_er(fig, axes[:, :], n_cmap_all, n_norm_all, n_k, N_unique, shrink=shrink)
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.savefig(base_path + save_prefix + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base_path + save_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_prefix}')

def expected_euler(n, p_arr):
    """
    Computes the closed-form expected Euler characteristic of an
    Erdos-Renyi G(n,p) clique complex: E[chi] = sum_{k=1}^{n}
    (-1)^{k-1} C(n,k) p^{C(k,2)}, both in full and truncated at k=4
    to match this module's TDA computation (tetrahedra = max simplex).

    Parameters
    ----------
    n : int
      Number of nodes.
    p_arr : ndarray
      Edge probabilities to evaluate at.

    Returns
    -------
    chi_full : ndarray
      Full expected Euler characteristic at each p in p_arr.
    chi_trunc : ndarray
      Expected Euler characteristic truncated to simplices of size
      <= 4, at each p in p_arr.
    """
    chi_full  = np.zeros_like(p_arr)
    chi_trunc = np.zeros_like(p_arr)
    for k in range(1, n + 1):
        term = (-1) ** (k - 1) * comb(n, k) * p_arr ** comb(k, 2)
        chi_full += term
        if k <= 4:
            chi_trunc += term
    return chi_full, chi_trunc

def add_discrete_cbar_ws(fig, ax_target, k_cmap_ws, norm_ws, n_k_ws, K_ws_graph, shrink=0.6):
    """
    Adds a discrete colorbar labeled by ring-lattice degree k to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
      Figure to attach the colorbar to.
    ax_target : Axes or array of Axes
      Axes the colorbar is placed relative to.
    k_cmap_ws : Colormap
      Colormap used for the k values.
    norm_ws : Normalize
      Normalization mapping k values to [0, 1].
    n_k_ws : int
      Number of discrete k values (tick count).
    K_ws_graph : list
      k values, in tick order, used as tick labels.
    shrink : float
      Colorbar shrink factor.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
      The created colorbar.
    """
    sm = cm.ScalarMappable(cmap=k_cmap_ws, norm=norm_ws)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_target, shrink=shrink, pad=0.02,
                        location='right', ticks=np.arange(n_k_ws))
    cbar.set_ticklabels([str(k) for k in K_ws_graph])
    cbar.set_label(r'$k$ (average degree)', fontsize=12)
    return cbar

def get_mean_traj_ws(data, n, k, iterations, field_fn):
    """
    Computes the mean, across iterations, of a field's trajectory for
    one Watts-Strogatz n/k combination, interpolated onto a common
    rewiring-probability grid spanning [0, 1].

    Parameters
    ----------
    data : dict
      Nested results dict indexed as data[n][k].
    n : int
      Node count to select.
    k : int
      Ring-lattice degree to select.
    iterations : iterable
      Run indices to average over.
    field_fn : callable
      Function (entry) -> 1-D trajectory, e.g. cR1_ws or cR2_ws.

    Returns
    -------
    x_common : ndarray or None
      Common rewiring-probability grid spanning [0, 1], or None if no
      iteration produced a valid trajectory.
    mean : ndarray or None
      Mean trajectory interpolated onto x_common, or None likewise.
    """
    trajs = []
    for iteration in iterations:
        entry = data[n][k].get(iteration, None)
        if entry is None:
            continue
        try:
            traj = field_fn(entry)
            if traj is not None:
                trajs.append(np.array(traj, dtype=float))
        except Exception:
            continue
    if not trajs:
        return None, None
    max_len  = max(len(t) for t in trajs)
    x_common = np.linspace(0, 1, max_len)
    interp   = np.array([
        np.interp(x_common, np.linspace(0, 1, len(t)), t)
        for t in trajs
    ])
    return x_common, interp.mean(axis=0)

def draw_ws_figure(data_ws_graph, fields_ws, n_plot, k_cmap_ws, norm_ws, n_k_ws, K_ws_graph,
                   k_colors_ws, k_ls_ws, iterations_ws_graph,
                   panel_layout, nrows, ncols, figsize,
                   suptitle, base_path, save_prefix, cbar_shrink=0.6):
    """
    Plots a grid of panels, each showing mean field trajectories vs.
    rewiring probability for the Watts-Strogatz model, one line per
    ring-lattice degree in K_ws_graph (colored and styled), and
    saves the figure.

    Parameters
    ----------
    data_ws_graph : dict
      Nested results dict indexed as data_ws_graph[n_plot][k] (see
      get_mean_traj_ws).
    fields_ws : dict
      Maps each panel label to a field_fn (see get_mean_traj_ws).
    n_plot : int
      Node count to select.
    k_cmap_ws : Colormap
      Colormap for the k colorbar.
    norm_ws : Normalize
      Normalization for the k colorbar.
    n_k_ws : int
      Number of discrete k values (colorbar tick count).
    K_ws_graph : list of int
      Ring-lattice degrees to plot, one line per value.
    k_colors_ws : dict
      Maps each k value to a plot color.
    k_ls_ws : dict
      Maps each k value to a linestyle.
    iterations_ws_graph : iterable
      Run indices to average over.
    panel_layout : list of str
      Panel labels (keys into fields_ws), in axes order.
    nrows : int
      Number of subplot rows.
    ncols : int
      Number of subplot columns.
    figsize : tuple of float
      Figure size passed to plt.subplots.
    suptitle : str
      Figure title.
    base_path : str
      Directory to save the figure files to.
    save_prefix : str
      Filename prefix (without extension) for the saved figure.
    cbar_shrink : float
      Colorbar shrink factor.

    Returns
    -------
    None
      Displays the figure and saves it as PDF and PNG under base_path.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes_arr  = np.array(axes).flatten()

    for ax, label in zip(axes_arr, panel_layout):
        field_fn = fields_ws[label]
        for k in K_ws_graph:
            x, mean = get_mean_traj_ws(data_ws_graph, n_plot, k, iterations_ws_graph, field_fn)
            if x is None:
                continue
            ax.plot(x, mean,
                    color=k_colors_ws[k],
                    lw=1.4,
                    linestyle=k_ls_ws[k])
        ax.set_ylabel(label)
        ax.set_xlabel(r'Rewiring probability $q$')
        ax.set_xlim(0, 1)

    for ax in axes_arr[len(panel_layout):]:
        ax.set_visible(False)

    add_discrete_cbar_ws(fig, axes_arr[:len(panel_layout)], k_cmap_ws, norm_ws, n_k_ws, K_ws_graph,
                         shrink=cbar_shrink)

    fig.suptitle(suptitle, fontsize=14, y=1.03)
    fig.savefig(base_path + save_prefix + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base_path + save_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_prefix}')

def get_value_at_q(data_ws_scaling, iterations_ws_scaling, n, k, field_fn, q_target):
    """
    Computes the mean, across iterations, of a field's value at one
    specific rewiring probability q_target, for one Watts-Strogatz
    n/k combination (interpolating each iteration's trajectory onto
    [0, 1] before sampling at q_target).

    Parameters
    ----------
    data_ws_scaling : dict
      Nested results dict indexed as data_ws_scaling[n][k].
    iterations_ws_scaling : iterable
      Run indices to average over.
    n : int
      Node count to select.
    k : int
      Ring-lattice degree to select.
    field_fn : callable
      Function (entry) -> 1-D trajectory.
    q_target : float
      Rewiring probability (in [0, 1]) to evaluate the field at.

    Returns
    -------
    float
      Mean value across iterations at q_target (NaN if none are valid).
    """
    vals = []
    for iteration in iterations_ws_scaling:
        entry = data_ws_scaling[n][k].get(iteration, None)
        if entry is None: continue
        try:
            traj = np.array(field_fn(entry), dtype=float)
            if len(traj) == 0: continue
            x = np.linspace(0, 1, len(traj))
            vals.append(np.interp(q_target, x, traj))
        except Exception: continue
    return np.mean(vals) if vals else np.nan

def add_discrete_colorbar_pa(fig, ax_target, k_cmap, norm, n_k, K, shrink = 0.6):
    """
    Adds a discrete colorbar labeled by attachment parameter k to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
      Figure to attach the colorbar to.
    ax_target : Axes or array of Axes
      Axes the colorbar is placed relative to.
    k_cmap : Colormap
      Colormap used for the k values.
    norm : Normalize
      Normalization mapping k values to [0, 1].
    n_k : int
      Number of discrete k values (tick count).
    K : list
      k values, in tick order, used as tick labels.
    shrink : float
      Colorbar shrink factor.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
      The created colorbar.
    """
    sm = cm.ScalarMappable(cmap=k_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_target, shrink=shrink, pad=0.02,
                         location='right',
                         ticks=np.arange(n_k))
    cbar.set_ticklabels([str(k) for k in K])
    cbar.set_label(r'$k$ (attachment parameter)', fontsize=12)
    return cbar

def get_trajectories(data, k, iterations, field_fn):
    """
    Collects one field trajectory per iteration for a fixed k,
    without averaging or interpolating.

    Parameters
    ----------
    data : dict
      Results dict indexed as data[k].
    k : object
      Key selecting the results cell (e.g. attachment parameter).
    iterations : iterable
      Run indices to collect trajectories for.
    field_fn : callable
      Function (entry) -> 1-D trajectory.

    Returns
    -------
    trajs : list of ndarray
      One trajectory per iteration that produced a valid (non-None)
      result.
    """
    trajs = []
    for iteration in iterations:
        entry = data[k].get(iteration, None)
        if entry is None:
            continue
        try:
            traj = field_fn(entry)
            if traj is not None:
                trajs.append(np.array(traj, dtype=float))
        except Exception:
            continue
    return trajs

def mean_traj(trajs):
    """
    Computes the mean of a list of trajectories (see get_trajectories),
    interpolated onto a common integer-step grid of length equal to
    the longest trajectory.

    Parameters
    ----------
    trajs : list of ndarray
      Trajectories to average, of possibly differing lengths.

    Returns
    -------
    x_common : ndarray or None
      Common step grid (0, 1, ..., max_len-1), or None if trajs is empty.
    mean : ndarray or None
      Mean trajectory interpolated onto x_common, or None likewise.
    """
    if not trajs:
        return None, None
    max_len  = max(len(t) for t in trajs)
    x_common = np.arange(max_len)
    interp   = np.array([
        np.interp(x_common, np.arange(len(t)), t)
        for t in trajs
    ])
    return x_common, interp.mean(axis=0)

def get_terminals(field_fn, data_dict, k_list, iterations):
    """
    Collects each iteration's terminal (final-timestep) field value,
    for each k in k_list.

    Parameters
    ----------
    field_fn : callable
      Function (entry) -> 1-D trajectory.
    data_dict : dict
      Results dict indexed as data_dict[k].
    k_list : iterable
      Keys (e.g. attachment parameters) to collect terminals for.
    iterations : iterable
      Run indices to collect terminal values for.

    Returns
    -------
    terminals : dict
      Maps each k in k_list to an ndarray of terminal values, one per
      iteration that produced a valid, non-empty trajectory.
    """
    terminals = {}
    for k in k_list:
        vals = []
        for iteration in iterations:
            entry = data_dict[k].get(iteration, None)
            if entry is None:
                continue
            try:
                traj = np.array(field_fn(entry), dtype=float)
                if len(traj) > 0:
                    vals.append(traj[-1])
            except Exception:
                continue
        terminals[k] = np.array(vals)
    return terminals

def clipped_yerr(means, stds, y_min=None, y_max=None):
    """
    Computes asymmetric error bars, clipping the lower/upper extent so
    means +/- error stays within [y_min, y_max].

    Parameters
    ----------
    means : ndarray
      Mean values at each point.
    stds : ndarray
      Standard deviations at each point (used as the unclipped error
      bar length in both directions).
    y_min : float or None
      Lower clip bound; if None, the lower error bar is not clipped.
    y_max : float or None
      Upper clip bound; if None, the upper error bar is not clipped.

    Returns
    -------
    ndarray, shape (2, n)
      Row 0 is the (clipped) lower error bar length, row 1 the
      (clipped) upper error bar length, for use as yerr in
      Axes.errorbar.
    """
    lower = stds.copy()
    upper = stds.copy()
    if y_min is not None:
        lower = np.minimum(lower, means - y_min)
    if y_max is not None:
        upper = np.minimum(upper, y_max - means)
    return np.array([lower, upper])

def plot_pa_ra(fields_pub, bounded_labels, data_ba, data_ra, K_shared, iterations,
               ba_color, ra_color, errbar_kw,
               ax, label, full_label = None):
    """
    Plots one field's terminal value vs. attachment parameter k, with
    error bars, for the Barabasi-Albert (PA) and random-attachment
    (RA) models overlaid on one axis.

    Parameters
    ----------
    fields_pub : dict
      Maps label to a field_fn (see get_terminals).
    bounded_labels : container of str
      Labels whose field is known to lie in [0, 1]; error bars for
      these are clipped to that range.
    data_ba : dict
      Barabasi-Albert results dict indexed as data_ba[k].
    data_ra : dict
      Random-attachment results dict indexed as data_ra[k].
    K_shared : list
      Attachment parameter values to plot on the x-axis.
    iterations : iterable
      Run indices to average over.
    ba_color : str
      Line/marker color for the PA (BA) series.
    ra_color : str
      Line/marker color for the RA series.
    errbar_kw : dict
      Extra keyword arguments passed to Axes.errorbar.
    ax : matplotlib.axes.Axes
      Axis to draw on.
    label : str
      Key into fields_pub selecting the field to plot; also used as
      the y-axis label unless full_label is given.
    full_label : str or None
      If given, used as the y-axis label instead of label.

    Returns
    -------
    None
      Draws onto ax and sets its labels/ticks.
    """
    field_fn  = fields_pub[label]
    is_bounded = label in bounded_labels
    y_min      = 0.0 if is_bounded else None
    y_max      = 1.0 if is_bounded else None

    term_ba = get_terminals(field_fn, data_ba,    K_shared, iterations)
    term_ra = get_terminals(field_fn, data_ra, K_shared, iterations)

    means_ba = np.array([term_ba[k].mean() if len(term_ba[k]) > 0
                         else np.nan for k in K_shared])
    stds_ba  = np.array([term_ba[k].std()  if len(term_ba[k]) > 1
                         else 0.0 for k in K_shared])
    means_ra = np.array([term_ra[k].mean() if len(term_ra[k]) > 0
                         else np.nan for k in K_shared])
    stds_ra  = np.array([term_ra[k].std()  if len(term_ra[k]) > 1
                         else 0.0 for k in K_shared])

    ax.errorbar(K_shared, means_ba,
                yerr=clipped_yerr(means_ba, stds_ba, y_min, y_max),
                fmt='o-', color=ba_color, ecolor=ba_color, **errbar_kw)
    ax.errorbar(K_shared, means_ra,
                yerr=clipped_yerr(means_ra, stds_ra, y_min, y_max),
                fmt='s--', color=ra_color, ecolor=ra_color, **errbar_kw)
    if not full_label:
      ax.set_ylabel(label)
    else:
      ax.set_ylabel(full_label)
    ax.set_xlabel(r'$k$ (attachment parameter)')
    ax.set_xticks(K_shared)

def get_mean_terminal(fields, fields_nlpa, iterations_graph, iterations_nlpa, k, alpha, label, data_nlpa, data_ra, data_ba):
    """
    Computes the mean terminal (final-timestep) field value for one
    k/alpha combination, dispatching to the RA data (alpha=0), BA
    data (alpha=1), or NLPA data (other alpha) as appropriate.

    Parameters
    ----------
    fields : dict
      Maps label to a field_fn for RA/BA entries (see get_terminals).
    fields_nlpa : dict
      Maps label to a field_fn for NLPA entries.
    iterations_graph : iterable
      Run indices to average over for RA/BA data.
    iterations_nlpa : iterable
      Run indices to average over for NLPA data.
    k : object
      Attachment parameter selecting the results cell.
    alpha : float
      Nonlinear preferential-attachment exponent; 0.0 selects RA
      data, 1.0 selects BA data, otherwise NLPA data.
    label : str
      Key into fields/fields_nlpa selecting the field to evaluate.
    data_nlpa : dict
      NLPA results dict indexed as data_nlpa[k][alpha].
    data_ra : dict
      Random-attachment results dict indexed as data_ra[k].
    data_ba : dict
      Barabasi-Albert results dict indexed as data_ba[k].

    Returns
    -------
    float
      Mean terminal value across the selected iterations (NaN if none
      are valid).
    """
    field_fn = fields[label]
    field_fn_nlpa = fields_nlpa[label]

    def extract(data_dict, iters, fn):
        """
        Computes the mean terminal field value across iterations for
        a single results dict (helper for the RA/BA branches above).

        Parameters
        ----------
        data_dict : dict
          Results dict indexed as data_dict[k].
        iters : iterable
          Run indices to average over.
        fn : callable
          Function (entry) -> 1-D trajectory.

        Returns
        -------
        float
          Mean terminal value across iterations (NaN if none are valid).
        """
        vals = []
        for it in iters:
            entry = data_dict[k].get(it, None)
            if entry is None: continue
            try:
                traj = np.array(fn(entry), dtype=float)
                if len(traj) > 0: vals.append(traj[-1])
            except Exception: continue
        return np.mean(vals) if vals else np.nan

    if alpha == 0.0:
        return extract(data_ra, iterations_graph, field_fn)
    elif alpha == 1.0:
        return extract(data_ba, iterations_graph, field_fn)
    else:
        vals = []
        for it in iterations_nlpa:
            entry = data_nlpa[k][alpha].get(it, None)
            if entry is None: continue
            try:
                traj = np.array(field_fn_nlpa(entry), dtype=float)
                if len(traj) > 0: vals.append(traj[-1])
            except Exception: continue
        return np.mean(vals) if vals else np.nan

def draw_nlpa_figure(data_ba, data_ra, data_nlpa, fields, fields_nlpa, iterations_graph, iterations_nlpa,
                      panels, nrows, ncols, figsize,
                     suptitle, base_path, save_prefix,
                     K_plot, k_styles, alphas_full):
    """
    Plots a grid of panels, each showing mean terminal field values vs.
    nonlinear preferential-attachment exponent alpha (spanning RA at
    alpha=0 through BA/PA at alpha=1, via get_mean_terminal), one line
    per attachment parameter in K_plot (colored/styled), with RA and
    PA reference lines annotated, and saves the figure.

    Parameters
    ----------
    data_ba : dict
      Barabasi-Albert results dict indexed as data_ba[k].
    data_ra : dict
      Random-attachment results dict indexed as data_ra[k].
    data_nlpa : dict
      NLPA results dict indexed as data_nlpa[k][alpha].
    fields : dict
      Maps panel label to a field_fn for RA/BA entries.
    fields_nlpa : dict
      Maps panel label to a field_fn for NLPA entries.
    iterations_graph : iterable
      Run indices to average over for RA/BA data.
    iterations_nlpa : iterable
      Run indices to average over for NLPA data.
    panels : list of list of str
      Grid of panel labels (keys into fields/fields_nlpa), row-major.
    nrows : int
      Number of subplot rows.
    ncols : int
      Number of subplot columns.
    figsize : tuple of float
      Figure size passed to plt.subplots.
    suptitle : str
      Figure title.
    base_path : str
      Directory to save the figure files to.
    save_prefix : str
      Filename prefix (without extension) for the saved figure.
    K_plot : list
      Attachment parameter values to plot, one line per value.
    k_styles : dict
      Maps each k in K_plot to a (color, marker, linestyle) tuple.
    alphas_full : list of float
      Alpha values plotted on the x-axis (must include 0.0 and 1.0).

    Returns
    -------
    None
      Displays the figure and saves it as PDF and PNG under base_path.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    all_labels = [label for row in panels for label in row]

    for ax, label in zip(axes_flat, all_labels):
        for k in K_plot:
            color, marker, ls = k_styles[k]
            means = [
                get_mean_terminal(fields, fields_nlpa, iterations_graph, iterations_nlpa, k, a, label, data_nlpa, data_ra, data_ba)
                for a in alphas_full
            ]
            ax.plot(alphas_full, means,
                    marker=marker, linestyle=ls,
                    color=color, lw=1.5, markersize=5)

        ax.axvline(x=0.0, color='#444444', lw=1.0,
                   linestyle='--', alpha=0.8, zorder=3)
        ax.axvline(x=1.0, color='#444444', lw=1.0,
                   linestyle='--', alpha=0.8, zorder=3)

        ymin, ymax = ax.get_ylim()
        ax.text(0.0, ymax, r'RA', fontsize=9, color='#444444',
                ha='center', va='bottom')
        ax.text(1.0, ymax, r'PA', fontsize=9, color='#444444',
                ha='center', va='bottom')

        ax.set_ylabel(label)
        ax.set_xlabel(r'$\alpha$ (preferential strength parameter)')
        ax.set_xticks(alphas_full)
        ax.set_xticklabels([str(a) for a in alphas_full],
                           rotation=45, ha='right')

    for ax in axes_flat[len(all_labels):]:
        ax.set_visible(False)

    handles = [
        mlines.Line2D([], [],
                      color=k_styles[k][0],
                      marker=k_styles[k][1],
                      linestyle=k_styles[k][2],
                      markersize=5,
                      label=rf'$k = {k}$')
        for k in K_plot
    ]
    fig.legend(handles=handles, loc='lower center',
               ncol=len(K_plot), frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.savefig(base_path + save_prefix + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base_path + save_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_prefix}')

# ============================================================================
# Classes
# ============================================================================

class WeightedSampler:
    """
    Efficient data structure for weighted sampling with dynamic updates
    for nonlinear preferential attachment model
    """

    def __init__(self, initial_weights):
        self.weights = list(initial_weights)
        self.cumulative = self._build_cumulative()
        self.total_weight = self.cumulative[-1] if self.cumulative else 0.0

    def _build_cumulative(self):
        """Build cumulative sum array for binary search"""
        cumulative = []
        total = 0.0
        for w in self.weights:
            total += w
            cumulative.append(total)
        return cumulative

    def update_weight(self, node_id, new_weight):
        """Update weight for a specific node"""
        if node_id >= len(self.weights):
            # Extend arrays for new nodes
            while len(self.weights) <= node_id:
                self.weights.append(0.0)
                self.cumulative.append(self.cumulative[-1] if self.cumulative else 0.0)

        old_weight = self.weights[node_id]
        weight_diff = new_weight - old_weight

        self.weights[node_id] = new_weight

        # Update cumulative sums from this point forward
        for i in range(node_id, len(self.cumulative)):
            self.cumulative[i] += weight_diff

        self.total_weight = self.cumulative[-1] if self.cumulative else 0.0

    def sample(self, exclude=None):
        """Sample a node using binary search on cumulative weights"""
        if exclude is None:
            exclude = set()

        if self.total_weight <= 0:
            return None

        while True:
            target = np.random.random() * self.total_weight
            idx = bisect.bisect_left(self.cumulative, target)
            if idx < len(self.weights) and idx not in exclude and self.weights[idx] > 0:
                return idx

        return None