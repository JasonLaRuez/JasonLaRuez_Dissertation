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
import matplotlib.pyplot as plt # Plotting
import time # Timing simulations
from tqdm.notebook import trange, tqdm # Allows for real-time progress bar of simulations

import sys
from rbloom import Bloom
import math
import seaborn as sns
import os
import gc # Memory management
import pickle # Takes environment variables and saves them as is
import gzip # Allows for compression of saved files
from joblib import Parallel, delayed # Parallelization functions
import multiprocessing # Get number of cpu cores
import collections

# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================
# Simplicial processing
# ============================================================================

def Get_Simpliciality_SF(filename, processed_filename = None, minDim = 2, maxDim = np.inf):
  """
  Input: filename = string file name to load hypergraph
         processed_filename = string file name to save data to
         minDim = minimum simplex size to consider. Simplices below this
                  dimension are disregarded. Simplices equal to this dimension
                  are automatically downward closed and now counted.
         maxDim = maximum simplex dimension to consider. Used to
                  reduce computational complexity for large hypergraphs
                  with high cardinality edges.
  Return: SimplicialFraction = ratio of downward-closed hyperedges to
                               the total number of hyperedges
          Lengths = dictionary keeping track of the number of simplices of
                    each dimension which are downward closed.
  Description: To compute the simplicial fraction we take a dynamic
  programming approach. We first sort the edges by ascending length.
  Then, to check if a simplex e is downward closed it is sufficient to
  check if its |e|-1 sized subsets belong to H and are downward closed.
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
    Input a networkx graph G and output the clique simplicial
    complex corresponding to G, where every node, edge and
    clique (up to and including size k) is encoded as simplices.
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
    Input a simplicial complex and integer n >= 0 and return all n-simplices (simplices with n+1 nodes)
    """
    # Filter function iterates through the faces of the complex and filters those with n+1 nodes
    return list(filter(lambda face: len(face) == n+1, Complex))

def SimplexCounts(Complex):
    """
    Input a simplical complex and return the counts of simplices of each dimension,
    as well as the dimension of the complex (largest dimension of simplices).
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
  Input a networkx graph G, and prime p to compute the betti
  numbers of the clique complex of G over Z mod p. Return the
  exact euler characteristic, as well as Betti numbers b0, b1, b2.
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
    Compute TDA quantities on the downward closure complex of hypergraph H.
    H: dict of {edge_id: tuple/set of node ids}
    Returns [b0, b1, b2, euler, counts] in the same format as ComputeResults.
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
    """O(1) removal from DiscordantEdges using swap-and-pop."""
    pos = DiscordantIndex.pop(edge_id)
    last = DiscordantEdges[-1]
    DiscordantEdges[pos] = last
    DiscordantEdges.pop()
    if DiscordantEdges:  # update index of moved element
        DiscordantIndex[last] = pos

def add_discordant(edge_id, DiscordantIndex, DiscordantEdges):
    """O(1) addition to DiscordantEdges."""
    DiscordantIndex[edge_id] = len(DiscordantEdges)
    DiscordantEdges.append(edge_id)

def has_disjoint_tuple(DiscordantEdges, E, source_neighbors, target_neighbors):
    # "any" function stops as soon as a True is found
    return any((E[t][0] not in source_neighbors and E[t][0] not in target_neighbors) and (E[t][1] not in source_neighbors and E[t][1] not in target_neighbors) for t in DiscordantEdges)

def G_RewireToRandomVoter(G, rho, alpha, exit_criteria = np.inf, timing = False):
    """
    Input a networkx object G, initial opinion 0 density rho, and
    rewiring probability alpha. Simulate the adaptive network voter
    model on the input graph, where at each step a discordant edge
    is selected uniformly. Then, with probability alpha the edge is rewired
    at random, and with probability 1-alpha one node adopts the opinion
    of its neighbor. Returns the proportions of opinions at each time step and
    the terminal graph G.
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
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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
    Input a networkx object G, initial opinion 0 density rho, and
    rewiring probability alpha. Simulate the adaptive network voter
    model on the input graph, where at each step a discordant edge
    is selected uniformly. Then, with probability alpha the edge is rewired
    to a node with the same opinion, and with probability 1-alpha one node adopts the opinion
    of its neighbor. Returns the proportions of opinions at each time step and
    the terminal graph G.
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
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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
    Input a networkx object G, initial opinion 0 density rho, and
    rewiring probability alpha. Simulate the adaptive network voter
    model on the input graph, where at each step a discordant edge
    is selected uniformly. Then, with probability alpha the edge is rewired
    at random, and with probability 1-alpha one node adopts the opinion
    of its neighbor. Returns the proportions of opinions at each time step and
    the terminal graph G.
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
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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
    return 1 if prop > 0.5 else (0 if prop < 0.5 else np.random.randint(2))

def proportional_vote(prop):
    return 1 if random.random() < prop else 0

def rewire_random(H, E, N, Opinions, Count, DiscordantIndex, DiscordantEdges, edge_id, edgeChoice):
    """
    Performs random rewiring by swapping nodes between edges.

    Input: H: Hypergraph dict
           E: List of edge ids
           N: Node neighborhood dict
           Opinions: Node opinion dict
           Prop: Edge proportion dict
           DiscordantEdges: List of discordant edges
           edge_id: Selected edge to rewire
           edgeChoice: Index of edge in DiscordantEdges
    Output: Updated data structures
    """
    # Select source node
    sourceEdge = H[edge_id]
    sourceNode_id = np.random.randint(len(sourceEdge))
    sourceNode = sourceEdge[sourceNode_id]

    while True:
        # Select random edge and check if valid
        targetEdge_id = random.choice(E)
        if targetEdge_id != edge_id:
            # Randomly select target node
            targetEdge = H[targetEdge_id]
            targetNode_id = np.random.randint(len(targetEdge))
            targetNode = targetEdge[targetNode_id]
            if (targetNode != sourceNode) and (sourceNode not in targetEdge) and (targetNode not in sourceEdge):
                break

    # Update original edge
    sourceEdge[sourceNode_id] = targetNode
    sourceEdge = sorted(sourceEdge)
    H[edge_id] = sourceEdge

    # Subtract contribution of source node and add contribution of target node
    Count[edge_id] = Count[edge_id] + Opinions[targetNode] - Opinions[sourceNode]

    if Count[edge_id] == 0 or Count[edge_id] == len(H[edge_id]):
        # Edge is now in consensus, remove from DiscordantEdges
        remove_discordant(edge_id, DiscordantIndex, DiscordantEdges)

    # Update targeted edge
    count = Count[targetEdge_id]
    targetEdge[targetNode_id] = sourceNode
    targetEdge = sorted(targetEdge)
    H[targetEdge_id] = targetEdge
    Count[targetEdge_id] = Count[targetEdge_id] + Opinions[sourceNode] - Opinions[targetNode]

    # Update DiscordantEdges for target edge
    if (count == 0 or count == len(targetEdge)) and (Count[targetEdge_id] != 0 and Count[targetEdge_id] != len(targetEdge)):
        add_discordant(targetEdge_id, DiscordantIndex, DiscordantEdges)
    elif (count != 0 and count != len(targetEdge)) and (Count[targetEdge_id] == 0 or Count[targetEdge_id] == len(targetEdge)):
        remove_discordant(targetEdge_id, DiscordantIndex, DiscordantEdges)

    # Update neighborhood of nodes
    N[sourceNode].remove(edge_id)
    N[sourceNode].add(targetEdge_id)
    N[targetNode].add(edge_id)
    N[targetNode].remove(targetEdge_id)


def social_influence(H, N, Opinions, Count, DiscordantIndex, DiscordantEdges, edge_id, edgeChoice, vote_function):
    """
    Performs social influence step where nodes adopt opinions based on voting rule.

    Input: H: Hypergraph dict
           N: Node neighborhood dict
           Opinions: Node opinion dict
           Prop: Edge proportion dict
           DiscordantEdges: List of discordant edges
           edge_id: Selected edge for influence
           edgeChoice: Index of edge in DiscordantEdges
           voting: Voting rule ('Majority' or 'Proportional')
    Output: Number of nodes that changed opinion and their new opinion
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


def HG_RewireRandom_Voter(H, alpha, rho, voting='Majority', exit_criteria = np.inf, timing=False):
    """
    Implementation of the hypergraph rewire-to-random model using
    a node-swapping rewiring rule.
    Input: Initial hypergraph dict H = {edge_id: edge_list}
           alpha: Probability of rewiring
           rho: Initial proportion of opinion 1 in H
           voting: Rule used for voting, either 'Majority' or 'Proportional'
           timing: Whether to print timing information
    Output: List of proportions of opinion 1 at each time step, and
            terminal hypergraph H.
    """

    vote_function = majority_vote if voting == 'Majority' else proportional_vote

    if timing:
        print("Initializing hypergraph, variables and data structures", flush=True)
        start = time.time()

    # Get node list by taking set union of edges
    V = set(v for e in H for v in H[e])
    # Store edge id list for random selection
    E = list(H.keys())
    # Create incidence dict to know what edges are incident to each node
    N = {v: set() for v in V}
    for e in H:
        for v in H[e]:
            N[v].add(e)

    # Set |V|*rho many nodes to opinion 1 and the rest to opinion 0
    Opinions = set(np.random.choice(list(V), size=round(rho*len(V)), replace=False))
    Opinions = {v: 1 if v in Opinions else 0 for v in V}

    # For each edge compute the proportion of member nodes with opinion 1
    Count = {e: sum(Opinions[v] for v in H[e]) for e in H}

    # Generate list of all edges where connected nodes have differing opinions
    DiscordantEdges = [e for e in H if Count[e] > 0 and Count[e] < len(H[e])]
    DiscordantIndex = {edge_id: pos for pos, edge_id in enumerate(DiscordantEdges)}

    # Initialize list of opinion proportions
    Proportions = [sum(Opinions.values()) / len(V)]
    DiscordantCounts = [len(DiscordantEdges) / len(E)]
    timer = 0

    if timing:
        end = time.time()
        print("Initialization complete, time taken: " + str(end - start) + " seconds", flush=True)
        print("Beginning network evolution", flush=True)
        start = time.time()

    while len(DiscordantEdges) > 0 and timer < exit_criteria:
        # Copy proportions from previous time step, and update timer
        timer += 1
        Proportions.append(Proportions[-1])

        # Uniformly select a discordant edge
        edgeChoice = np.random.randint(len(DiscordantEdges))
        edge_id = DiscordantEdges[edgeChoice]

        # Random Rewiring (probability alpha)
        if random.random() < alpha:
            rewire_random(H, E, N, Opinions, Count, DiscordantIndex, DiscordantEdges, edge_id, edgeChoice)

        # Social influence (probability 1-alpha)
        else:
            num_changed, new_opinion = social_influence(H, N, Opinions, Count,
                                                        DiscordantIndex, DiscordantEdges, edge_id,
                                                        edgeChoice, vote_function)

            # Update global proportion
            if new_opinion == 1:
                Proportions[timer] += num_changed / len(V)
            else:
                Proportions[timer] -= num_changed / len(V)

        DiscordantCounts.append(len(DiscordantEdges) / len(E))

    if timing:
        end = time.time()
        print("Evolution complete, time taken: " + str(end - start) + " seconds", flush=True)

    return np.array(Proportions), np.array(DiscordantCounts), H

def PC_HG_RewireRandom(params):
  """
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
  """
  # Grab parameter values from params list
  n = params[0]; m = params[1]; rho = params[2]; alpha = params[3]; voting = params[4]; iteration = params[5];
  # Create filename from params
  filename =  'Voter/Triangle/G_rewire_triangle_'+str(n)+'_'+str(m)+'_'+str(rho).replace('.','_')+'_'+str(alpha).replace('.','_')+'_'+voting+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
    return 0

  G = nx.gnm_random_graph(n,m)
  Proportions, DiscordantCounts, H = HG_RewireRandom_Voter(H, alpha, rho, voting=voting)
  data = [Proportions, DiscordantCounts, H]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def HG_RewireSame_Voter(H, alpha, rho, voting='Majority',
                         exit_criteria=np.inf, timing=False):

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
    Hypergraph voter model combining structure-aware transitivity enforcement
    with opinion-aware rewire-to-same as fallback.

    With probability alpha, rewiring occurs:
      - With probability gamma: a triangle-closing swap is attempted,
        selecting nodes purely on structural grounds (no opinion filtering).
        If no valid triangle swap exists, falls back to rewire-to-same.
      - With probability 1-gamma (or on triangle failure): rewire-to-same
        is used, swapping minority-opinion nodes between edges of opposite
        majority to drive consensus.
    With probability 1-alpha, social influence is applied.
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
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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

  Proportions, DiscordantCounts, H = HG_RewireTriangleSame_Voter(H, alpha, rho, gamma, voting=voting)
  data = [Proportions, DiscordantCounts, H]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def PC_Simplicial_WS(params):
  """
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
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
    """Generate k-uniform ER hypergraph"""
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
    Run a single simulation for given parameters
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
    Loads a single trajectory file and returns (Proportions, DiscordantCounts).
    Returns (None, None) for missing or corrupted files.
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
    Create 3D arch plot for graph model (K=2)
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

def plot_er_terminal(er_means, n, m, rhos, TICK_IDX, XLABELS, YLABELS, save_path=None):
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
    rs = str(rho).replace('.', '_')
    return os.path.join(PROC_G,
        f'tda_rho{rs}_a{alpha:.4f}_g{gamma:.4f}_it{iteration}.pkl.gz')

def compute_and_save_graph(n, m, rho, alpha, gamma, iteration, BASE_G, PROC_G):
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
    c = counts_g[rho]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, sums_g[rho][fld] / c, np.nan)

def plot_graph_heatmaps_styled(derived_g, CMAP, TICK_IDX, XLABELS, YLABELS, n, m, rho, save_path=None):

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
    """Path to the individual terminal graph file."""
    return os.path.join(
        BASE_GRAPH, model,
        f'G_{model}_{n}_{m}_'+str(rho).replace('.','_')+'_'+str(alpha).replace('.','_')+'_'+str(gamma).replace('.','_')+'_'+str(iteration)+'.pkl'
    )

def proc_fname_baws(BASE_GRAPH, model, alpha, gamma, iteration):
    """Path to the saved TDA result for one (model, alpha, gamma, iteration)."""
    proc_path = os.path.join(BASE_GRAPH, model, 'proc')
    os.makedirs(proc_path, exist_ok=True)
    return os.path.join(proc_path, f'tda_a{alpha:.4f}_g{gamma:.4f}_it{iteration}.pkl.gz')

def load_raw_graph(path):
    """Try gzip pickle first, fall back to plain pickle."""
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
    c = counts_baws[model]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, sums_baws[model][fld] / c, np.nan)

def plot_baws_heatmaps(n, m, rho, derived_baws, CMAP, TICK_IDX, XLABELS, YLABELS, model, save_path=None):
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

def compute_and_save_HG(k, alphas, gammas, ai, gi, it):
    pfname = proc_fname(k, ai, gi, it)
    if os.path.isfile(pfname):
        return 'exists'
    rpath = raw_fname(k, alphas[ai], gammas[gi], it)
    if not os.path.isfile(rpath):
        return 'missing'
    try:
        with gzip.open(rpath, 'rb') as f:
            data = pickle.load(f)

        proportion = float(data[0])
        H          = data[3]

        tda  = ComputeHGResults(H)          # ← downward closure, not clique
        cnts = tda[4] if tda[4] is not None else []

        es_result = Get_Simpliciality_ES(H, minDim=2, maxDim=np.inf)
        es        = float(es_result[0])
        sf_result = Get_Simpliciality_SF(H, minDim=2, maxDim=np.inf)
        sf        = float(sf_result[0])

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

def plot_simplicial_heatmaps(n, rho, derived, CMAP, XLABELS, YLABELS, TICK_A, TICK_G,
                             k, psub, save_path=None):
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