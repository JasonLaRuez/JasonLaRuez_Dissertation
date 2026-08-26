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
from itertools import combinations, product # For getting different simplices and all combinations of lists

import matplotlib.lines as mlines
import matplotlib.cm as cm
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
    Incremental average clustering
    Adding edge (u,v) only affects the local clustering of u, v, and
    their common neighbors. Recompute only those nodes using nx.clustering,
    updating the running sum by subtracting old values and adding new ones.
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
  Input integer n = number of nodes in ER graph and terminal edge density
  0 <= p <= 1. Iteratively adds  random edges to ER graph until the graph
  is complete or until the added edge density surpasses p. Returns
  the sets of the 0-th, 1-st, 2-nd and 3-rd betti numbers using
  persistent homology (3-rd betti number is truncated betti number),
  the counts of different dimensional simplices at each time step,
  the set of Euler characteristics, the cycle rank at each time step,
  and the average clustering coefficient at each time step.
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
  Input integer n = terminal number of nodes in BA graph, and
  positive integer k which is the number of nodes that a newly added
  node is connected to when it is added to the graph. Returns
  the sets of the 0-th, 1-st, 2-nd and 3-rd betti numbers using
  persistent homology (3-rd betti number is truncated betti number),
  the counts of different dimensional simplices at each time step,
  and the set of Euler characteristics.
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
  Input integer n = terminal number of nodes in graph, and
  positive integer k which is the number of nodes that a newly added
  node is connected to when it is added to the graph. Returns
  the sets of the 0-th, 1-st, 2-nd and 3-rd betti numbers using
  persistent homology (3-rd betti number is truncated betti number),
  the counts of different dimensional simplices at each time step,
  and the set of Euler characteristics.
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
    print(nx.average_clustering(G), clusterC[timer])

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
  Input integer n = number of nodes in WS graph, and
  even integer m which is the degree of each node
  in the initial ring lattice. Returns the sets of the
  0-th, 1-st, 2-nd and 3-rd betti numbers using zigzag
  persistent homology (3-rd betti number is truncated betti number),
  the counts of different dimensional simplices at each time step,
  and the set of Euler characteristics.
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
  Input integer n = terminal number of nodes in BA graph, and
  positive integer k which is the number of nodes that a newly added
  node is connected to when it is added to the graph. alpha is the
  preferential strength parameter in nonlinear preferential attachment. 
  Returns the sets of the 0-th, 1-st, 2-nd and 3-rd betti numbers using
  persistent homology (3-rd betti number is truncated betti number),
  the counts of different dimensional simplices at each time step,
  and the set of Euler characteristics.
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
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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
    Load a single file and return (k, iteration, data).
    Returns None for the data field if file is missing or corrupted.
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
    sc = entry[1]
    return np.array([
        row[dim] if len(row) > dim else 0
        for row in sc
    ], dtype=float)

def cR1(e):
    """Stored cycleR = N1 - N0 + beta0"""
    return np.array(e[3], dtype=float)

def cR2(e):
    return sc_col(e, 2) - cR1(e) + e[0][1].astype(float)

def cR1_nlpa(e):
    return np.array(e[4], dtype=float)

def cR2_nlpa(e):
    return sc_col(e, 2) - cR1_nlpa(e) + e[0][1].astype(float)

def sc_col_ws(sc_list, dim):
    if sc_list is None:
        return np.zeros(1)
    return np.array([
        row[dim] if (row is not None and len(row) > dim) else 0
        for row in sc_list
    ], dtype=float)

def cR1_ws(e):
    return (sc_col_ws(e[1], 1)
            - sc_col_ws(e[1], 0)
            + e[0][0].astype(float))

def cR2_ws(e):
    return sc_col_ws(e[1], 2) - cR1_ws(e) + e[0][1].astype(float)

def get_mean_traj_er(data, n, p, iterations, field_fn):
    """
    Mean trajectory over iterations, x normalised to [0, p].
    Returns (x_common, mean) or (None, None) if no data.
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
    sm = cm.ScalarMappable(cmap=n_cmap, norm=n_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_target, shrink=shrink, pad=0.02,
                         location='right', ticks=np.arange(n_n))
    cbar.set_ticklabels([str(n) for n in N])
    cbar.set_label(r'$N$ (number of nodes)', fontsize=12)
    return cbar

def draw_er_figure(data_er, n_colors_all, NP_sorted, fields_er_pub, iterations, n_cmap_all, n_norm_all, n_k, N_unique, panel_layout, suptitle, base_path, save_prefix, shrink=0.5):
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

def expected_Nk(n, k, p):
    """Expected number of k-simplices (k-cliques) in G(n,p)."""
    return comb(n, k) * p ** comb(k, 2)

def expected_euler(n, p_arr):
    """
    E[chi] = sum_{k=1}^{n} (-1)^{k-1} C(n,k) p^{C(k,2)}
    Truncated at k=4 to match TDA computation (tetrahedra = max simplex).
    Both full and truncated versions returned.
    """
    chi_full  = np.zeros_like(p_arr)
    chi_trunc = np.zeros_like(p_arr)
    for k in range(1, n + 1):
        term = (-1) ** (k - 1) * comb(n, k) * p_arr ** comb(k, 2)
        chi_full += term
        if k <= 4:
            chi_trunc += term
    return chi_full, chi_trunc

def expected_simplex_counts(n, p_arr, max_k=4):
    """Return E[N_k] for k = 1, ..., max_k."""
    return {k: comb(n, k) * p_arr ** comb(k, 2) for k in range(1, max_k + 1)}

def add_discrete_cbar_ws(fig, ax_target, k_cmap_ws, norm_ws, n_k_ws, K_ws_graph, shrink=0.6):
    sm = cm.ScalarMappable(cmap=k_cmap_ws, norm=norm_ws)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_target, shrink=shrink, pad=0.02,
                        location='right', ticks=np.arange(n_k_ws))
    cbar.set_ticklabels([str(k) for k in K_ws_graph])
    cbar.set_label(r'$k$ (average degree)', fontsize=12)
    return cbar

def get_mean_traj_ws(data, n, k, iterations, field_fn):
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
    sm = cm.ScalarMappable(cmap=k_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_target, shrink=shrink, pad=0.02,
                         location='right',
                         ticks=np.arange(n_k))
    cbar.set_ticklabels([str(k) for k in K])
    cbar.set_label(r'$k$ (attachment parameter)', fontsize=12)
    return cbar

def get_trajectories(data, k, iterations, field_fn):
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
    Compute asymmetric error bars clipped to [y_min, y_max].
    Returns array of shape (2, n) for use as yerr in errorbar.
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
    field_fn = fields[label]
    field_fn_nlpa = fields_nlpa[label]

    def extract(data_dict, iters, fn):
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