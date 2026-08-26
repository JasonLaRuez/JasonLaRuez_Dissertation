"""
Module Name: HypergraphModels.py
Description: Contains functions for generating ER, WS and PA hypergraphs,
then computing topological quantities (Betti #'s, EC, simplexcounts)

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

import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.special import gammaln
from scipy import stats
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt # Plotting
import time # Timing simulations
from tqdm.notebook import trange, tqdm # Allows for real-time progress bar of simulations

from rbloom import Bloom
import sys
import gc # Memory management
import os
import pickle # Takes environment variables and saves them as is
import gzip # Allows for compression of saved files
from concurrent.futures import ProcessPoolExecutor # Parallelization functions
from joblib import Parallel, delayed # Parallelization functions
import multiprocessing # Get number of cpu cores
import collections # Collecting degree dist using counter

import xgi
import math
from collections import defaultdict
from array import array
import bisect

# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================
# Simplicial/Hypergraph Processing
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

def Get_Simpliciality_LSF(filename, processed_filename, minDim = 2, maxDim = np.inf):
  """
  Input: filename = string file name to load hypergraph
         processed_filename = string file name to save data to
         minDim = minimum simplex size to consider. Simplices below this
                  dimension are disregarded. Simplices equal to this dimension
                  are automatically downward closed and now counted.
         maxDim = maximum simplex dimension to consider. Used to
                  reduce computational complexity for large hypergraphs
                  with high cardinality edges.
  Return: LSF = list of tuple pairs (LSF(v), deg(v)) for all vertices v in H
          Assortativity = assortativity of LSF
  Description: First, we determine whether each edge in H is downward closed
  using the same method as Get_Simpliciality_SF. Then for each node v we form
  the neighborhood N(v). At that point we simply check whether or not each
  each edge in N(v) is downward closed in H, since if a subset of e belongs
  to N(v) then any subsets of e in H must also belong to N(v).
  """
  # Load data from input filename
  with gzip.open(filename, 'rb') as f:
    try:
      with gzip.open(filename, 'rb') as f:
        H, _ = pickle.load(f)

    except Exception as e:
      print(f"Error loading file: {filename}")
      return -1

  # Create a list of (key, set) tuples sorted by ascending set length
  sorted_items = sorted(H.items(), key=lambda item: len(item[1]), reverse=False)

  # Create a new dictionary with keys 0, 1, 2, ... corresponding to sorted sets
  H = {i: tuple(sorted(item[1])) for i, item in enumerate(sorted_items)}
  del(sorted_items); gc.collect()

  # Initialize dict that keeps track of edges in H, as well as whether
  # or not they are downward closed (True or False).
  IsSimplex = {e: True for e in H.values()}

  # Here we label each edge as downward closed or not
  for t in tqdm(range(len(H))):
    # If the hyperedge is below the minimum size we discard it from consideration.
    # If hyperedge is at minimum size it is automatically a simplex, and so it is
    # disregarded from simpliciality computation
    e = H[t]

    if (len(e) <= minDim) or (len(e) > maxDim):
      continue

    # Iterate over the subsets of e of size |e|-1. e is downward closed
    # iff its |e|-1 subsets belong to the hypergraph, and are all downward
    # closed themselves.
    for face in combinations(e, len(e)-1):
      if (face not in IsSimplex) or (not IsSimplex[face]):
        IsSimplex[e] = False
        break

  # Replace H with H_v = {v: [e | v in e]}. This lets us iterate over the
  # vertices and their neighborhoods rather than over hyperedges.
  H_v = {}
  for t in H:
    for v in H[t]:
      if v not in H_v:
        # (adjacent edges, adjacent nodes, simpliciality, degree)
        H_v[v] = [[H[t]], None, None, None]
      else:
        H_v[v][0].append(H[t])
  # Cleanup original hypergraph to free up space
  del(H); gc.collect()

  # Each timestep represents a single node and its neighborhood
  for v in tqdm(H_v):
    # Get set of nodes in neighborhood
    N_v = set(u for e in H_v[v][0] for u in e)
    # Get set of edges in the neighborhood (exclude edges smaller than minDim, and edges larger than maxDim)
    Edges = set(e for u in N_v for e in H_v[u][0]
                if ( (len(e) >= minDim) and (len(e) <= maxDim) and set(e).issubset(N_v)) )
    Edges = sorted(Edges, key=lambda item: len(item), reverse=False)
    # If v is an isolated node, or all edges in its neighborhood are not larger than
    # the minimum simplex size we disregard the node entirely
    if (len(Edges) == 0) or ( len(Edges[-1]) <= minDim ):
      # Leave None in place of simpliciality and adjacent nodes
      # since we are disregarding the node, and record its degree
      H_v[v][3] = len(H_v[v])

    else:
      # Keep track of num nodes which are downward closed and total number of
      # hyperedges with |e| > m
      SimplexCount = 0
      CandidateCount = 0

      for e in Edges:
        # If |e| = minDim disregard it
        if len(e) == minDim:
          continue
        # e is a simplex so increase candidate and simplex counts
        elif IsSimplex[e]:
          SimplexCount += 1; CandidateCount += 1;
        # e is NOT a simplex so don't increase candidate count
        else:
          CandidateCount += 1

      # For each node with a valid neighborhood we record the adjacent nodes,
      # the SimplicialFraction of its neighborhood as well as the degree of the node
      H_v[v][1] = N_v
      H_v[v][2] = SimplexCount / CandidateCount
      H_v[v][3] = len(H_v[v][0])

  # Compute mean and variance of LSF across all nodes
  # Consider only nodes with valid node neighborhoods
  V = [v for v in H_v if H_v[v][2] != None]
  LSF = [H_v[v][2] for v in V]
  mean = np.mean(LSF); var = np.var(LSF); Assortativity = 0

  # Iterate over all vertices pairwise to compute the simplicial assortativity
  numPairs = 0

  # If the mean LSF is 0 then all LSF's are 0 so we skip this computation
  if mean > 0:
    # Iterate over all valid vertices
    for i in range(len(V)):
      v = V[i]
      # Iterate over the neighbors of v
      for u in H_v[v][1]:
        # Ensure we don't double compute, and check that u has a valid neighborhood
        if (u < v) and (H_v[u][2] != None):
          numPairs += 1
          Assortativity += (H_v[u][2] - mean) * (H_v[v][2] - mean) / var

    Assortativity /= numPairs

  else:
    Assortativity = np.nan

  # We will return pairs (LSF(v), deg(v))
  LSF = [(H_v[v][2], H_v[v][3]) for v in H_v]
  Data = [LSF, Assortativity]

  # Pickle and save the output
  with gzip.open(processed_filename, 'wb') as f:
    pickle.dump(Data, f)

  return LSF, Assortativity

def process_batch_of_nodes(batch_data):
    vertices, H_v_batch, IsSimplex, minDim, maxDim = batch_data
    results = {}

    for v in vertices:
        node_edges = H_v_batch[v][0]

        # Get set of node neighbors of v for assortativity computation
        N_v = set(u for e in node_edges for u in e)

        # Get set of edges in the neighborhood
        Edges = set(e for u in N_v for e in H_v_batch[u][0]
                    if ((len(e) >= minDim) and (len(e) <= maxDim) and set(e).issubset(N_v)))
        Edges = sorted(Edges, key=lambda item: len(item), reverse=False)

        # Initialize result with default values
        result = [node_edges, None, None, None]

        # If v is an isolated node, or all edges in its neighborhood are not larger than
        # the minimum simplex size we disregard the node entirely
        if (len(Edges) == 0) or (len(Edges[-1]) <= minDim):
            result[3] = len(node_edges)
        else:
            # Track simplex counts
            SimplexCount = 0
            CandidateCount = 0

            for e in Edges:
                if len(e) == minDim:
                    continue
                elif IsSimplex[e]:
                    SimplexCount += 1
                    CandidateCount += 1
                else:
                    CandidateCount += 1

            result[1] = N_v
            result[2] = SimplexCount / CandidateCount if CandidateCount > 0 else 0
            result[3] = len(node_edges)

        results[v] = result

    return results

def create_batches(items, batch_count):
    """
    Distribute items to workers in a strided pattern
    where worker j gets vertices i*num_workers + j
    """
    all_vertices = list(items)
    batches = [[] for _ in range(batch_count)]

    # Assign vertices in a strided pattern
    for i, vertex in enumerate(all_vertices):
        worker_index = i % batch_count
        batches[worker_index].append(vertex)

    return batches

def PC_Get_Simpliciality_LSF(filename, processed_filename, numWorkers = multiprocessing.cpu_count(), minDim = 2, maxDim = np.inf):
  """
  Input: filename = string file name to load hypergraph
         processed_filename = string file name to save data to
         minDim = minimum simplex size to consider. Simplices below this
                  dimension are disregarded. Simplices equal to this dimension
                  are automatically downward closed and now counted.
         maxDim = maximum simplex dimension to consider. Used to
                  reduce computational complexity for large hypergraphs
                  with high cardinality edges.
  """

  # Load data from input filename
  with gzip.open(filename, 'rb') as f:
    try:
      with gzip.open(filename, 'rb') as f:
        H, _ = pickle.load(f)

    except Exception as e:
      print(f"Error loading file: {filename}")
      return -1

  # Create a list of (key, set) tuples sorted by ascending set length
  sorted_items = sorted(H.items(), key=lambda item: len(item[1]), reverse=False)

  # Create a new dictionary with keys 0, 1, 2, ... corresponding to sorted sets
  H = {i: tuple(sorted(item[1])) for i, item in enumerate(sorted_items)}
  del(sorted_items); gc.collect()

  # Initialize dict that keeps track of edges in H, as well as whether
  # or not they are downward closed (True or False).
  IsSimplex = {e: True for e in H.values()}

  # Each timestep represents the addition of a single hyperedge
  for t in tqdm(range(len(H))):
    # To reduce memory overusage we cleanup every so often
    if t > 0 and t % 500000 == 0:
      gc.collect()

    # If the hyperedge is below the minimum size we discard it from consideration.
    # If hyperedge is at minimum size it is automatically a simplex, and so it is
    # disregarded from simpliciality computation
    e = H[t]

    if (len(e) <= minDim) or (len(e) > maxDim):
      continue

    # Iterate over the subsets of e of size |e|-1. e is downward closed
    # iff its |e|-1 subsets belong to the hypergraph, and are all downward
    # closed themselves.
    for face in combinations(e, len(e)-1):
      if (face not in IsSimplex) or (not IsSimplex[face]):
        IsSimplex[e] = False
        break

  # Replace H with H_v = {v: [e | v in e]}. This lets us iterate over the
  # vertices and their neighborhoods rather than over hyperedges.
  H_v = {}
  for t in H:
    for v in H[t]:
      if v not in H_v:
        # (adjacent edges, adjacent nodes, simpliciality, degree)
        H_v[v] = [[H[t]], None, None, None]
      else:
        H_v[v][0].append(H[t])
  # Cleanup original hypergraph to free up space
  del(H); gc.collect()

  # Determine number of workers and create batches
  all_vertices = list(H_v.keys())
  batches = create_batches(all_vertices, numWorkers)

  # Create a shared dict for IsSimplex
  manager = multiprocessing.Manager()
  shared_IsSimplex = manager.dict(IsSimplex)
  shared_H_v = manager.dict(H_v)

  # Prepare the arguments for each batch
  batch_args = [(batch, shared_H_v, shared_IsSimplex, minDim, maxDim) for batch in batches]

  # Process batches in parallel
  results_dict = {}
  with ProcessPoolExecutor(max_workers=numWorkers) as executor:
      batch_results = list(tqdm(
          executor.map(process_batch_of_nodes, batch_args),
          total=len(batch_args),
          desc="Processing vertex batches"
      ))

      # Combine results from all batches
      for batch_result in batch_results:
          results_dict.update(batch_result)

  # Update H_v with the results
  for v, result in results_dict.items():
      H_v[v] = result


  # Compute mean and variance of LSF across all nodes
  V = [v for v in H_v if H_v[v][2] != None]
  LSF = [H_v[v][2] for v in V]
  mean = np.mean(LSF); var = np.var(LSF); Assortativity = 0

  # Iterate over all vertices pairwise to compute the simplicial assortativity
  numPairs = 0

  # If the mean LSF is 0 then all LSF's are 0 so we skip this computation
  if mean > 0:
    for i in range(len(V)):
      v = V[i]
      for u in H_v[v][1]:
        if (u < v) and (H_v[u][2] != None):
          numPairs += 1
          Assortativity += (H_v[u][2] - mean) * (H_v[v][2] - mean) / var

    Assortativity /= numPairs

  else:
    Assortativity = np.nan

  # We will return pairs (LSF(v), deg(v))
  LSF = [(H_v[v][2], H_v[v][3]) for v in H_v]
  Data = [LSF, Assortativity]

  # Pickle and save the output
  with gzip.open(processed_filename, 'wb') as f:
    pickle.dump(Data, f)

  return 0

def SimplexCounts(Complex):
    """
    Input a simplical complex and return the counts of simplices of each dimension,
    as well as the dimension of the complex (largest dimension of simplices).
    """
    if not Complex:
        return []

    max_size = max(len(t) for t in Complex)
    Counts = [0] * max_size

    for t in tqdm(Complex):
        Counts[len(t) - 1] += 1

    return Counts

def DownwardClosureComplex(H, k = float('inf')):
    """
    Input a networkx graph G and output the clique simplicial
    complex corresponding to G, where every node, edge and
    clique (up to and including size k) is encoded as simplices.
    """
    Complex = set()

    # Iterate across each maximal clique, and include each subset of the clique
    # as a simplex in the complex. The set() structure, with sorted(), avoids the issue of
    # including identical simplices multiple times.
    for e in tqdm(H.values()):
        numNodes = len(e)
        # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
        # all the way down to the 2-subsets (edges) and 1-subsets (nodes)
        # We do sorted() to avoid double creating simplices
        e = sorted(e)
        for r in range(min(numNodes, k), 0, -1):
            # combinations(S,r) is from itertools and returns iterator corresponding
            # to all r-subsets of S.
            for face in combinations(e, r):
                Complex.add(face)

    return (Complex)

def ComputeResults(filename, processed_filename, p = 2):
  """
  Input a networkx graph G, and prime p to compute the betti
  numbers of the clique complex of G over Z mod p. Return the
  exact euler characteristic, as well as Betti numbers b0, b1, b2.
  """
  if type(filename) == dict:
      H = filename
  else:
    with gzip.open(filename, 'rb') as f:
      try:
        with gzip.open(filename, 'rb') as f:
          H, _ = pickle.load(f)

      except Exception as e:
        print(f"Error loading file: {filename}")
        return -1

  print("Constructing Complex")
  Complex = DownwardClosureComplex(H, k = 4)
  Counts = SimplexCounts(Complex); Euler = 0
  for i in range(len(Counts)):
    Euler += Counts[i] * (-1)**i

  Simplices = [(list(simplex), len(simplex)-1) for simplex in Complex if len(simplex) <= 4]

  print("Compute Persistence")
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

  data = [Betti[0][-1], Betti[1][-1], Betti[2][-1], Euler, Counts]
  # Pickle and save the output
  with gzip.open(processed_filename, 'wb') as f:
      pickle.dump(data, f)

  return data

def Get_SimplexCounts_EulerChar(filename, processed_filename, maxMemory, maxDim = np.inf):
  # Load data from input filename
  with gzip.open(filename, 'rb') as f:
    try:
      with gzip.open(filename, 'rb') as f:
        H, _ = pickle.load(f)

    except Exception as e:
      print(f"Error loading file: {filename}")
      return -1

  # Get list of time indexes at which hyperedges were added
  T = list(H.keys())
  # Get the max simplex size, which is either the largest hyperedge size
  # in H, or a set limit MaxDim
  maxSize = min(max(map(len, H.values())),maxDim)
  # Initialize massive array to keep track of simplex counts
  SimplexCounts = np.zeros((len(T), maxSize))
  # Initialize array to store Euler Characteristic
  Euler = np.zeros(len(T))
  
  # Initialize a bloom-filter to store the simplices. This is a
  # probabilistic set which is incredibly memory efficient with the tradeoff
  # of introducing minor error. Input is max size of the set() and the error rate
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
  for t in T:
    e_len = len(H[t])
    if e_len > maxDim:
      continue
    # Calculate the number of potential missing simplices using combinations
    num_potential_missing = 0
    # Iterate over sizes from minDim to e_len - 1
    for r in range(1, e_len):
        num_potential_missing += math.comb(e_len, r)

    maxFilterSize += num_potential_missing

    if maxFilterSize >= max_elements:
      maxFilterSize = max_elements
      break

  # Ensure maxFilterSize is at least 1 to avoid an error when creating the Bloom filter with size 0.
  # This can happen if T becomes 0 because the first edge is too large or has insufficient dimensions.
  maxFilterSize = max(1, maxFilterSize)
  Simplices = Bloom(maxFilterSize,0.005)

  # Each timestep represents the addition of a single hyperedge
  counter = 0
  for t in tqdm(T):
    if len(H[t]) > maxDim:
      continue
    e = sorted(H[t])

    if counter > 0:
      SimplexCounts[counter] = SimplexCounts[counter-1].copy()
    for r in range(len(e), 0, -1):
      # combinations(S,r) is from itertools and returns iterator corresponding
      # to all r-subsets of S.
      for face in combinations(e, r):
        if face not in Simplices:
          SimplexCounts[counter][r-1] += 1
          Simplices.add(face)
    # Compute the Euler Characteristic as the alternating sum of
    # p-simplex counts
    for i in range(maxSize):
      Euler[counter] += SimplexCounts[counter][i] * ( (-1) ** i )
    counter += 1

  # Pickle and save the output
  Data = [SimplexCounts[:counter,:4], Euler]
  with gzip.open(processed_filename, 'wb') as f:
    pickle.dump(Data, f)

  return 0

def check_if_simplex(Subsets, Supersets, IsSimplex, SimplexCount, e):
  """
  Recursive function
  """
  # By default increase simplex count by 1.
  IsSimplex[e] = True
  SimplexCount += 1;

  # Iterate over the subsets of e of size |e|-1. e is downward closed
  # iff its |e|-1 subsets belong to the hypergraph, and are all downward
  # closed themselves.
  for face in combinations(e, len(e)-1):
      if (frozenset(face) not in Subsets[e]) or (not IsSimplex[frozenset(face)]):
          # Since a subset is not a simplex, e is not a simplex
          IsSimplex[e] = False; SimplexCount -= 1
          return SimplexCount

  # If e is a simplex we now need to check if adding e made any of its |e|+1
  # size subsets a simplex, and if so, recursively check +1 subsets for closure
  for superset in Supersets[e]:
    if not IsSimplex[superset]:
      SimplexCount = check_if_simplex(Subsets, Supersets, IsSimplex, SimplexCount, superset)

  return SimplexCount


def recheck_if_simplex(Subsets, Supersets, IsSimplex, SimplexCount, e):
    """
    Recheck if e is still a simplex after a subset was removed.
    Similar to check_if_simplex but for updates.
    """
    IsSimplex[e] = False
    SimplexCount -= 1

    for superset in Supersets[e]:
        if IsSimplex[superset]:
            SimplexCount = recheck_if_simplex(Subsets, Supersets, IsSimplex, SimplexCount, superset)

    return SimplexCount


def Get_Simpliciality_SF_TS_step(H_frozen, t, SimplexCount, CandidateCount, subsets, supersets, Subsets, Supersets, IsSimplex, E, minDim = 2, maxDim = np.inf):
    """

    """

    e = H_frozen[t]

    # If edge is below or above min/max simplex size, disregard
    if (len(e) < minDim) or (len(e) > maxDim) or (e in E):
        return SimplexCount, CandidateCount
    # If edge e is min simplex size, it is a simplex but dont count it towards SF.
    # For each |e|+1 size superset of e, check if adding e made the superset a simplex
    elif len(e) == minDim:
        IsSimplex[e] = True
        local_supersets = {H_frozen[superset_idx] for superset_idx in supersets if len(H_frozen[superset_idx]) == len(e) + 1}
        Supersets[e].update(local_supersets)
        for superset in local_supersets:
            Subsets[superset].add(e)
            # If a superset isn't already a simplex, check if adding e made it a simplex
            if not IsSimplex[superset]:
                SimplexCount = check_if_simplex(Subsets, Supersets, IsSimplex, SimplexCount, superset)
    # Otherwise we need to check if e is a simplex
    else:
        CandidateCount += 1
        local_subsets = {H_frozen[subset_idx] for subset_idx in subsets if (len(H_frozen[subset_idx]) == len(e) - 1)}
        Subsets[e].update(local_subsets)
        for subset in local_subsets:
            Supersets[subset].add(e)
        local_supersets = {H_frozen[superset_idx] for superset_idx in supersets if len(H_frozen[superset_idx]) == len(e) + 1}
        Supersets[e].update(local_supersets)
        for superset in local_supersets:
            Subsets[superset].add(e)
        # Check if e is a simplex, and recursively check if supersets are simplices as well
        SimplexCount = check_if_simplex(Subsets, Supersets, IsSimplex, SimplexCount, e)

    E.add(e)

    return SimplexCount, CandidateCount

def Remove_Simpliciality_SF_TS_step(H_frozen, t, SimplexCount, CandidateCount, subsets, supersets, Subsets, Supersets, IsSimplex, E, minDim = 2, maxDim = np.inf):
    """
    Remove an edge and update SF-related data structures.
    """
    e = H_frozen[t]

    # If edge is not in E or outside bounds, nothing to remove
    if (len(e) < minDim) or (len(e) > maxDim) or (e not in E):
        return SimplexCount, CandidateCount

    local_subsets = {H_frozen[subset_idx] for subset_idx in subsets if (len(H_frozen[subset_idx]) == len(e) - 1)}
    local_supersets = {H_frozen[superset_idx] for superset_idx in supersets if len(H_frozen[superset_idx]) == len(e) + 1}

    # Remove from E
    E.remove(e)

    # If edge was a candidate (not minDim)
    if len(e) > minDim:
        if IsSimplex[e]:
            SimplexCount -= 1
        CandidateCount -= 1
        IsSimplex[e] = False

    # Remove from subset/superset relationships
    for subset in local_subsets:
        Supersets[subset].remove(e)

    for superset in local_supersets:
        Subsets[superset].remove(e)
        # Recheck if superset is still a simplex after removing e
        if IsSimplex[superset]:
            SimplexCount = recheck_if_simplex(Subsets, Supersets, IsSimplex, SimplexCount, superset)

    # Clear this edge's relationships
    Subsets[e].clear()
    Supersets[e].clear()

    return SimplexCount, CandidateCount

def Get_Simpliciality_FES_TS_step(H_frozen, t, subsets, supersets, AllSubsets, AllSupersets, IsMaximal, minDim = 2, maxDim = np.inf):
    """

    """

    e = H_frozen[t]

    # If edge is below or above min/max simplex size, disregard
    if (len(e) < minDim) or (len(e) > maxDim):
        pass
    # If edge is min simplex size we dont include as a face in FES computation
    # but still add to subsets for any present supersets
    elif len(e) == minDim:
        AllSupersets[e].update( {H_frozen[superset_idx] for superset_idx in supersets})
        for superset_idx in supersets:
            AllSubsets[H_frozen[superset_idx]].add(e)
    # Check if the newly added edge is maximal, if so add it to IsMaximal
    else:
        AllSubsets[e].update( {H_frozen[subset_idx] for subset_idx in subsets} )
        for subset_idx in subsets:
            AllSupersets[H_frozen[subset_idx]].add(e)
        if len(supersets) == 0:
            IsMaximal.add(e)
            # All subsets of a maximal edge are not maximal
            for subset in AllSubsets[e]:
                if subset in IsMaximal:
                    IsMaximal.remove(subset)
        # If not maximal, then add e to subsets of its supersets (which exist since it is not maximal)
        else:
            AllSupersets[e].update( {H_frozen[superset_idx] for superset_idx in supersets})
            for superset_idx in supersets:
                AllSubsets[H_frozen[superset_idx]].add(e)

    FES = 0
    for face in IsMaximal:
        num_disregard = 0
        for i in range(1,minDim):
            num_disregard += math.comb(len(face),i)
        FES += (len(AllSubsets[face]) + 1) / ( 2 ** (len(face)) - 1 - num_disregard )

    return ( FES / len(IsMaximal) if len(IsMaximal) > 0 else 0 )


def Remove_Simpliciality_FES_TS_step(H_frozen, t, subsets, supersets, AllSubsets, AllSupersets, IsMaximal, minDim = 2, maxDim = np.inf):
    """
    Remove an edge and update FES-related data structures.
    """
    e = H_frozen[t]

    # If edge is below or above min/max simplex size, disregard
    if (len(e) < minDim) or (len(e) > maxDim):
        pass
    elif len(e) == minDim:
        # Remove e from supersets' AllSubsets
        for superset_idx in supersets:
            AllSubsets[H_frozen[superset_idx]].remove(e)
    else:
        for subset_idx in subsets:
            AllSupersets[H_frozen[subset_idx]].remove(e)
        # If e was maximal, remove it from IsMaximal
        if e in IsMaximal:
            IsMaximal.remove(e)
            # Check if any of e's subsets should become maximal
            for subset in AllSubsets[e]:
                # A subset becomes maximal if it has no supersets in the current hypergraph
                if subset not in IsMaximal and len(AllSupersets[subset]) == 0:
                    IsMaximal.add(subset)
        else:
            # Remove e from supersets' AllSubsets
            for superset_idx in supersets:
                AllSubsets[H_frozen[superset_idx]].remove(e)

    # Clear e's AllSubsets
    AllSubsets[e].clear()
    AllSupersets[e].clear()

def simpliciality_process_hypergraph(H):
    """
    Process a hypergraph dictionary incrementally, handling both additions and removals.

    Args:
        H: dict mapping timestep -> set of nodes
           Removals are indicated by negative timesteps in the dict

    Returns:
        List of results for each timestep
    """
    processor = HypergraphProcessor()

    # Find all timesteps (both positive and negative)
    all_timesteps = sorted(set(abs(t) for t in H.keys()))

    # If tqdm is available, use it; otherwise fall back to regular iteration
    try:
        from tqdm import tqdm
        iterator = tqdm(all_timesteps)
    except ImportError:
        iterator = all_timesteps

    upper_pairs = []
    lower_pairs = []

    # Initialize dict that keeps track of edges in H, as well as whether
    # or not they are downward closed (True or False).
    # Convert hyperedges to frozensets so they can be used as dictionary keys
    H_frozen = {t: frozenset(edge) for t, edge in H.items()}
    IsSimplex = {e: False for e in H_frozen.values()}
    Subsets = {e: set() for e in H_frozen.values()}
    Supersets = {e: set() for e in H_frozen.values()}
    SF_ts = []
    E = set()

    # Keep track of nodes which are downward closed and total number of
    # hyperedges with |e| > minDim (respectively)
    SimplexCount = 0
    CandidateCount = 0

    # Keep a set of all maximal edges, and for each edge all subsets of the edge
    # added so far.
    IsMaximal = set()
    AllSubsets = {e: set() for e in H_frozen.values()}
    AllSupersets = {e: set() for e in H_frozen.values()}
    FES_ts = []

    for t in iterator:
        upper_pairs_removed = 0; lower_pairs_removed = 0;
        # First check if there's a removal at timestep -t
        if -t in H_frozen and t != 0:
            # Process removal
            timestep_rem, sorted_edge_rem, subsets_rem, supersets_rem, upper_pairs_removed, lower_pairs_removed = processor.remove_edge_and_check(-t, H_frozen[-t], H_frozen)

            # Update SF data structures
            SimplexCount, CandidateCount = Remove_Simpliciality_SF_TS_step(
                H_frozen, -t, SimplexCount, CandidateCount,
                subsets_rem, supersets_rem, Subsets, Supersets, IsSimplex, E
            )

            # Update FES data structures
            Remove_Simpliciality_FES_TS_step(
                H_frozen, -t, subsets_rem, supersets_rem, AllSubsets, AllSupersets, IsMaximal
            )

        # Now process addition at timestep t
        if t in H_frozen:
            timestep, sorted_edge, subsets, supersets = processor.add_edge_and_check(t, H_frozen[t], H_frozen)
            upper_pairs.append(len(subsets)-upper_pairs_removed)
            lower_pairs.append(len(supersets)-lower_pairs_removed)

            SimplexCount, CandidateCount = Get_Simpliciality_SF_TS_step(
                H_frozen, t, SimplexCount, CandidateCount,
                subsets, supersets, Subsets, Supersets, IsSimplex, E
            )
            SF_ts.append( SimplexCount / CandidateCount if CandidateCount > 0 else 0 )

            fes = Get_Simpliciality_FES_TS_step(
                H_frozen, t, subsets, supersets, AllSubsets, AllSupersets, IsMaximal
            )
            FES_ts.append(fes)

    return (np.array(upper_pairs), np.array(lower_pairs), np.array(SF_ts), np.array(FES_ts))

def PH_add_e(Simplices, SimplexCounts, Times, e, timer):
    # Add e to simplices, update simplex counts, and add to times
    Simplices.add(e)
    SimplexCounts[timer][len(e)-1] += 1
    # By including simplices up to order 3, we ensure
    # b0, b1 and b2 are accurate
    if len(e) <= 4:
        Times.append((list(e), timer))
    # Next we need to repeat this process for any non-empty subsets
    # which were not previously added as simplices
    for size in range(2, len(e)):
        for subset in combinations(e, size):
            if subset not in Simplices:
                Simplices.add(subset)
                SimplexCounts[timer][len(subset)-1] += 1
                # By including simplices up to order 3, we ensure
                # b0, b1 and b2 are accurate
                if len(subset) <= 4:
                    Times.append((list(subset), timer))

def PH(SimplexCounts, start, Times, timer, timing):
    # Create filtration of simplicial complexes using Times
    f = d.Filtration(Times)

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
        # Only compute b0, b1, b2, b3
        if i < 4:
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

    if timing:
        end = time.time()
        print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds",flush=True)
    
    return Betti, Euler

def Extract_XtYt(H, timing = False):
  H_working = H.copy()

  # Initialize tracking variables
  iteration = 0
  isolated_nodes_per_iteration = []
  removed_edges_info = []

  # Store initial degrees of all nodes for efficient updates
  node_degrees = {}
  temp_degrees = H_working.nodes.degree.aslist(); i = 0
  for node in H_working.nodes:
      node_degrees[node] = temp_degrees[i]
      i += 1

  print("Starting iterative edge removal process...")
  print(f"Initial hypergraph: {H_working.num_nodes} nodes, {H_working.num_edges} edges")
  print(f"Computed initial degrees for {len(node_degrees)} nodes")

  # Find all edges with valid timestamps
  edges_with_timestamps = []
  for edge_id in H_working.edges:
      timestamp = H_working.edges[edge_id].get('timestamp')
      if timestamp is not None:
          edge_size = H_working.edges.size[edge_id]
          edges_with_timestamps.append((edge_id, timestamp, edge_size))
  # Sort by timestamps (ascending) then iterate backwards
  edges_with_timestamps.sort(key=lambda x: x[1])

  while H_working.num_edges > 0:
      # Find the latest timestamp (last element since sorted)
      latest_timestamp = edges_with_timestamps[-1][1]

      # Find all edges with the latest timestamp by scanning backwards
      latest_edges = []
      i = len(edges_with_timestamps) - 1
      while i >= 0 and edges_with_timestamps[i][1] == latest_timestamp:
          latest_edges.append(edges_with_timestamps[i])
          i -= 1
      # Sort by edge size
      latest_edges.sort(key=lambda x: x[2])

      # Remove the latest edges from edges_with_timestamps (they're at the end)
      edges_with_timestamps = edges_with_timestamps[:i+1]

      while len(latest_edges) > 0:

        # Find the largest edge among those with the latest timestamp
        # Breaking ties arbitrarily by taking the first one after sorting by edge_id
        largest_edge = latest_edges[-1]
        edge_id_to_remove, timestamp, edge_size = largest_edge
        latest_edges.remove(largest_edge)

        # Get the nodes in the edge before removal
        nodes_in_edge = list(H_working.edges.members(edge_id_to_remove))

        # Remove the edge
        H_working.remove_edge(edge_id_to_remove)

        for node in nodes_in_edge:
          if node in node_degrees:
              node_degrees[node] -= 1

        # Check how many nodes became isolated (degree 0)
        isolated_count = 0
        isolated_nodes = []

        for node in nodes_in_edge:
          if node in node_degrees:  # Check if node still exists in our degree tracking
              if node_degrees[node] == 0:
                  isolated_count += 1
                  isolated_nodes.append(node)

        # Record the results
        isolated_nodes_per_iteration.append(isolated_count)
        removed_edges_info.append({
          'iteration': iteration,
          'edge_id': edge_id_to_remove,
          'timestamp': timestamp,
          'edge_size': edge_size,
          'isolated_nodes': isolated_count,
          'nodes_in_edge': nodes_in_edge,
          'isolated_node_ids': isolated_nodes
        })

        # Print progress
        if (iteration % 100 == 0 or iteration < 10) and (timing == True):
          print(f"Iteration {iteration}: Removed edge {edge_id_to_remove} "
                f"(timestamp: {timestamp}, size: {edge_size}), "
                f"isolated nodes: {isolated_count}")

        iteration += 1

  print(f"\nCompleted {iteration} iterations")
  print(f"Final hypergraph: {H_working.num_nodes} nodes, {H_working.num_edges} edges")

  # Analysis of results
  print("\n=== ANALYSIS RESULTS ===")
  print(f"Total edges removed: {len(removed_edges_info)}")
  print(f"Total isolated nodes across all iterations: {sum(isolated_nodes_per_iteration)}")
  print(f"Average isolated nodes per iteration: {np.mean(isolated_nodes_per_iteration):.2f}")
  print(f"Maximum isolated nodes in single iteration: {max(isolated_nodes_per_iteration)}")

  # Find iterations with highest isolation
  max_isolation = max(isolated_nodes_per_iteration)
  max_isolation_iterations = [i for i, count in enumerate(isolated_nodes_per_iteration) if count == max_isolation]

  print(f"\nIterations with maximum isolation ({max_isolation} nodes):")
  for iter_idx in max_isolation_iterations[:5]:  # Show first 5
      info = removed_edges_info[iter_idx]
      print(f"  Iteration {iter_idx}: Edge {info['edge_id']} "
            f"(timestamp: {info['timestamp']}, size: {info['edge_size']})")

  # Visualize the results
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

  # Plot 1: Isolated nodes per iteration
  ax1.plot(range(len(isolated_nodes_per_iteration)), isolated_nodes_per_iteration,
           'b-', alpha=0.7, linewidth=1)
  ax1.fill_between(range(len(isolated_nodes_per_iteration)), isolated_nodes_per_iteration,
                   alpha=0.3, color='blue')
  ax1.set_xlabel('Iteration')
  ax1.set_ylabel('Number of Isolated Nodes')
  ax1.set_title('Nodes Becoming Isolated After Each Edge Removal')
  ax1.grid(True, alpha=0.3)

  # Plot 2: Cumulative isolated nodes
  cumulative_isolated = np.cumsum(isolated_nodes_per_iteration)
  ax2.plot(range(len(cumulative_isolated)), cumulative_isolated,
           'r-', linewidth=2)
  ax2.set_xlabel('Iteration')
  ax2.set_ylabel('Cumulative Isolated Nodes')
  ax2.set_title('Cumulative Number of Isolated Nodes')
  ax2.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show()

  # Summary statistics
  print(f"\n=== SUMMARY STATISTICS ===")
  print(f"Iterations with 0 isolated nodes: {isolated_nodes_per_iteration.count(0)}")
  print(f"Iterations with 1 isolated node: {isolated_nodes_per_iteration.count(1)}")
  print(f"Iterations with >1 isolated nodes: {sum(1 for x in isolated_nodes_per_iteration if x > 1)}")

  # Show edge size distribution of removed edges
  edge_sizes = [info['edge_size'] for info in removed_edges_info]
  print(f"\nRemoved edge sizes - Min: {min(edge_sizes)}, Max: {max(edge_sizes)}, Mean: {np.mean(edge_sizes):.2f}")

  return removed_edges_info

def process_dataset(name, timing=False):
    filename = 'XtYt_FromData/' + str(name) + '.pkl'
    # Check if current file already exists, if it does exist we do not
    # want to waste time running it again
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping...")
        return

    H = xgi.load_xgi_data(name)
    H = H.cleanup(multiedges=True, singletons=False, isolates=True, relabel=False, in_place=True)

    extracted_info = Extract_XtYt(H, timing=timing)
    # Pickle and save the output
    with gzip.open(filename, 'wb') as f:
        pickle.dump(extracted_info, f)
    print(f"Saved {filename}")

# ============================================================================
# Generative Models
# ============================================================================

def HG_ErdosRenyi_kUnif(n, K, p = 1, timing = False):
    """
    Creates an evolving k-uniform Erdos-Renyi model on hypergraphs by starting
    with a hypergraph with n nodes and no edges. Then, each edge
    of size K is selected uniformly at random and added.
    Input: int n, the number of nodes in H
           int K, the edge sizes
           float p, the proportion of edges to add
           bool timing, whether to display timing information
    Output: Betti, an array of the 0, 1 and 2 Betti numbers for each time step
            SimplexCounts
    """
    if timing:
        print("(1/5) Initializing hypergraph, variables and data structures",flush=True)
        start = time.time()

    nodes = range(n)

    # Generate a list of all edges of size = K that can be formed from n nodes
    E = [e for e in combinations(nodes, K)]
    # By shuffing this list, it is equivalent to forming a filtration where
    # starting from an empty Hgraph, edges are selected u.a.r and added
    random.shuffle(E)
    # Use set structure to remember what simplices have been added
    Simplices = set(tuple([v]) for v in nodes)

    # Initialize array of simplices and times they were added to complex
    Times = []

    # Add vertex simplices
    timer = 0
    for v in nodes:
      Times.append(([v], 0))

    # Intialize simplex counts with n vertices
    # The maximum simplex dimension is either n-1 or maxDim
    SimplexCounts = np.zeros( (len(E)+1 , min(K, n)) )
    SimplexCounts[0][0] = n

    if timing:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    for e in tqdm(E) if timing else E:
        timer += 1; SimplexCounts[timer] = SimplexCounts[timer-1].copy();
        
        # If yes then e was added as a subset in a previous step
        if e in Simplices:
            continue
        else:
            PH_add_e(Simplices, SimplexCounts, Times, e, timer)

        # Early stopping condition
        if timer > p * len(E):
            SimplexCounts = SimplexCounts[:timer+1]
            break

    if timing:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Persistent Homology",flush=True)
    start = time.time()

    Betti, Euler = PH(SimplexCounts, start, Times, timer, timing)

    H = {i: e for i, e in enumerate(E) if i < timer}

    return H, Betti, SimplexCounts, Euler

def HG_ErdosRenyi(n, K, p = 1, timing = False):
    """
    Creates an evolving Erdos-Renyi model on hypergraphs by starting
    with a hypergraph with n nodes and no edges. Then, each edge
    of size <= K is selected uniformly at random and added.
    Input: int n, the number of nodes in H
           int K, the maximum allowable edge size
           float p, the proportion of edges to add
           bool timing, whether to display timing information
    Output: Betti, an array of the 0, 1 and 2 Betti numbers for each time step
            SimplexCounts
    """
    if timing:
        print("(1/5) Initializing hypergraph, variables and data structures",flush=True)
        start = time.time()

    nodes = range(n)

    # Generate a list of all edges of size <= K that can be formed from n nodes
    E = [e for size in range(2, K+1) for e in combinations(nodes, size)]
    # By shuffing this list, it is equivalent to forming a filtration where
    # starting from an empty Hgraph, edges are selected u.a.r and add
    random.shuffle(E)
    # Use set structure to remember what simplices have been added
    Simplices = set(tuple([v]) for v in nodes)

    # Initialize array of simplices and times they were added to complex
    Times = []

    # Add vertex simplices
    timer = 0
    for v in nodes:
      Times.append(([v], 0))

    # Intialize simplex counts with n vertices
    # The maximum simplex dimension is either n-1 or maxDim
    SimplexCounts = np.zeros( (len(E)+1 , min(K, n)) )
    SimplexCounts[0][0] = n
    # Initialize edge counts
    EdgeCounts = np.zeros( (len(E)+1 , min(K, n)) )
    EdgeCounts[0][0] = n

    if timing:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    for e in tqdm(E) if timing else E:
        timer += 1; SimplexCounts[timer] = SimplexCounts[timer-1].copy();
        EdgeCounts[timer] = EdgeCounts[timer-1].copy(); EdgeCounts[timer][len(e)-1] += 1
        
        # If yes then e was added as a subset in a previous step
        if e in Simplices:
            continue
        else:
            PH_add_e(Simplices, SimplexCounts, Times, e, timer)

        # Early stopping condition
        if timer > p * len(E):
            SimplexCounts = SimplexCounts[:timer+1]
            break

    if timing:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Persistent Homology",flush=True)
    start = time.time()

    Betti, Euler = PH(SimplexCounts, start, Times, timer, timing)

    # Explicitly construct the hypergraph dictionary for simpliciality processing
    H = {i: e for i, e in enumerate(E) if i < timer}

    return H, Betti, SimplexCounts, EdgeCounts, Euler

def HG_PreferentialAttachment_kUnif(K, steps, timing=False):
    """
    Input:
    Output: Hypergraph dictionary H along with the degree frequency distribution D.
    """
    if timing:
        print("(1/5) Initializing hypergraph, variables and data structures",flush=True)
        start = time.time()

    # Initialize node list V, and hypergraph as a dictionary
    # Start with m nodes, add 1 node at each step, so at the end
    # there will "steps" many nodes
    H = {0:tuple([i for i in range(K-1)])}
    Simplices = set(subset for size in range(1,len(H[0])+1) for subset in combinations(H[0], size))

    # Keep track of vertex degrees for degree distribution
    D = [1 if i < K-1 else 0 for i in range(steps)]
    repeated_nodes = [i for i in range(K-1)];

    # Initialize array of simplices and times they were added to complex
    Times = []
    timer = 0

    for simplex in Simplices:
        Times.append((list(simplex), 0))

    # Intialize simplex counts with n vertices
    # The maximum simplex dimension is either n-1 or maxDim
    SimplexCounts = np.zeros( (steps + 1 - (K-1), min(K, steps)) )
    SimplexCounts[0][0] = K-1

    if timing:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    for step in (tqdm(range(K-1,steps)) if timing else range(K-1,steps)):
        timer += 1; SimplexCounts[timer] = SimplexCounts[timer-1].copy();
        # Select m many targets in graph using preferential attachment
        targets = set()
        while len(targets) < K-1:
            x = random.choice(repeated_nodes)
            targets.add(x)
        targets = list(targets)
        # Add the m many new nodes to the new hyperedge
        H_edge = tuple(targets + [step])
        H[timer] = H_edge
        repeated_nodes.extend(H_edge)

        # Update node degrees
        for v in H_edge: D[v] += 1

        Simplices.add(tuple([step])); SimplexCounts[timer][0] += 1
        Times.append((list([step]), timer))
        PH_add_e(Simplices, SimplexCounts, Times, H_edge, timer)

    if timing:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Persistent Homology",flush=True)
    start = time.time()

    Betti, Euler = PH(SimplexCounts, start, Times, timer, timing)

    # Returns D as a dictionary with degree values as keys and
    # the number of nodes with the given degree as entries.
    D =  collections.Counter(D)

    return H, D, Betti, SimplexCounts, Euler

def HG_PreferentialAttachment_Simplicial(K, p, steps, timing=False):
    """

    """
    if timing:
        print("(1/5) Initializing hypergraph, variables and data structures",flush=True)
        start = time.time()

    # Initialize node list V, and hypergraph as a dictionary
    # Start with m nodes, add 1 node at each step, so at the end
    # there will "steps" many nodes
    H = {0:tuple([i for i in range(K-1)])}
    E = set([tuple([i for i in range(K-1)])])
    Simplices = set(subset for size in range(1,len(H[0])+1) for subset in combinations(H[0], size))

    # Keep track of vertex degrees for degree distribution
    D = [1 if i < K-1 else 0 for i in range(steps)]
    repeated_nodes = [i for i in range(K-1)];

    # Initialize array of simplices and times they were added to complex
    Times = []
    timer = 0; edgeCounter = 1

    for simplex in Simplices:
        Times.append((list(simplex), 0))

    # Intialize simplex counts with n vertices
    # The maximum simplex dimension is either n-1 or maxDim
    SimplexCounts = np.zeros( (steps + 1 - (K-1), min(K, steps)) )
    SimplexCounts[0][0] = K-1

    if timing:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    for step in (tqdm(range(K-1,steps)) if timing else range(K-1,steps)):
        timer += 1; SimplexCounts[timer] = SimplexCounts[timer-1].copy();
        # Select m many targets in graph using preferential attachment
        targets = set()
        while len(targets) < K-1:
            x = random.choice(repeated_nodes)
            targets.add(x)
        targets = list(targets)
        # Add the m many new nodes to the new hyperedge
        H_edge = tuple(targets + [step])

        for size in range(2, len(H_edge)):
            for subset in combinations(H_edge, size):
                if subset not in E and random.random() < p:
                    E.add(subset)
                    H[edgeCounter] = subset; edgeCounter += 1
                    for v in subset: D[v] += 1
                    repeated_nodes.extend(subset)

        H[edgeCounter] = H_edge; edgeCounter += 1
        E.add(H_edge)
        repeated_nodes.extend(H_edge)

        # Update node degrees
        for v in H_edge: D[v] += 1

        Simplices.add(tuple([step])); SimplexCounts[timer][0] += 1
        Times.append((list([step]), timer))
        PH_add_e(Simplices, SimplexCounts, Times, H_edge, timer)

    if timing:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Persistent Homology",flush=True)
    start = time.time()


    Betti, Euler = PH(SimplexCounts, start, Times, timer, timing)

    # Returns D as a dictionary with degree values as keys and
    # the number of nodes with the given degree as entries.
    D =  collections.Counter(D)

    return H, D, Betti, SimplexCounts, Euler

def HG_NL_PreferentialAttachment_kUnif(K, steps, alpha = 1.0, timing=False):
    """
    Input:
    Output: Hypergraph dictionary H along with the degree frequency distribution D.
    """
    if timing:
        print("(1/5) Initializing hypergraph, variables and data structures",flush=True)
        start = time.time()

    # Initialize node list V, and hypergraph as a dictionary
    # Start with m nodes, add 1 node at each step, so at the end
    # there will "steps" many nodes
    H = {0:tuple([i for i in range(K-1)])}
    Simplices = set(subset for size in range(1,len(H[0])+1) for subset in combinations(H[0], size))

    # Keep track of vertex degrees for degree distribution
    D = [1 if i < K-1 else 0 for i in range(steps)]
    
    # Initialize efficient weighted sampler
    initial_weights = [1.0] * (K-1)
    sampler = WeightedSampler(initial_weights)

    # Initialize array of simplices and times they were added to complex
    Times = []
    timer = 0

    for simplex in Simplices:
        Times.append((list(simplex), 0))

    # Intialize simplex counts with n vertices
    # The maximum simplex dimension is either n-1 or maxDim
    SimplexCounts = np.zeros( (steps + 1 - (K-1), min(K, steps)) )
    SimplexCounts[0][0] = K-1

    if timing:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    for step in (tqdm(range(K-1,steps)) if timing else range(K-1,steps)):
        timer += 1; SimplexCounts[timer] = SimplexCounts[timer-1].copy();
        # Select m many targets in graph using preferential attachment
        targets = set()
        while len(targets) < K-1:
            x = sampler.sample(exclude=targets)
            if x is not None:
                targets.add(x)
            else:
                break
        targets = list(targets)
        # Add the m many new nodes to the new hyperedge
        H_edge = tuple(targets + [step])
        H[timer] = H_edge
    

        # Batch update degrees and weights
        for v in H_edge:
            D[v] += 1
            new_degree = D[v]
            new_weight = new_degree ** alpha
            sampler.update_weight(v, new_weight)

        Simplices.add(tuple([step])); SimplexCounts[timer][0] += 1
        Times.append((list([step]), timer))
        PH_add_e(Simplices, SimplexCounts, Times, H_edge, timer)

    if timing:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Persistent Homology",flush=True)
    start = time.time()

    Betti, Euler = PH(SimplexCounts, start, Times, timer, timing)

    # Returns D as a dictionary with degree values as keys and
    # the number of nodes with the given degree as entries.
    D =  collections.Counter(D)

    return H, D, Betti, SimplexCounts, Euler

def HG_WattsStrogatz_kUnif(n, K, timing = False):
    """
    ...
    Input: int n, the number of nodes in H
           int K, the maximum allowable edge size
           bool timing, whether to display timing information
    Output: Betti, an array of the 0, 1 and 2 Betti numbers for each time step
            SimplexCounts
    """
    if timing:
        print("(1/5) Initializing hypergraph, variables and data structures",flush=True)
        start = time.time()

    nodes = range(n)

    # Generate a list of all edges of size <= K that can be formed from n node ring lattice
    E = [tuple(sorted([(v+i) % n for i in range(K)])) for v in nodes]
    Edges = set()
    Edges.update(E)
    H = {i: e for i, e in enumerate(E)}
    # By shuffing this list, it is equivalent to forming a filtration where
    # edges are rewired u.a.r
    random.shuffle(E)

    # Use set structure to remember what simplices have been added
    Simplices = set(tuple([v]) for v in nodes)
    # Intialize simplex counts with n vertices
    # The maximum simplex dimension is either n-1 or maxDim
    SimplexCounts = np.zeros( (len(E)+1 , min(K, n)) )
    SimplexCounts[0][0] = n

    # Initialize dictionary which keeps track of when simplices
    # are added and removed
    Times = {tuple([v]) : [0] for v in nodes}
    timer = 0
    SuperSets = dict()

    # Need to keep track of how many supersets there are for the initial
    # simplices, as these are the only simplices that can be removed, which
    # only happens when 1. they are not an edge and 2. are not contained within
    # any other edges.
    for e in E:
        Simplices.add(e)
        SimplexCounts[0][len(e)-1] += 1
        Times[e] = [0]
        for size in range(2, len(e)):
            for subset in combinations(e, size):
                if subset not in SuperSets:
                    SuperSets[subset] = 1
                    Simplices.add(subset)
                    SimplexCounts[0][len(subset)-1] += 1
                    Times[subset] = [0]
                else:
                    SuperSets[subset] += 1

    if timing:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    for e in tqdm(E) if timing else E:
        timer += 1; SimplexCounts[timer] = SimplexCounts[timer-1].copy();
        # e is being rewired so we remove it from edges
        H[-(timer + len(E) - 1)] = e
        Edges.remove(e)
        Simplices.remove(e)
        SimplexCounts[timer][len(e)-1] -= 1
        Times[e].append(timer)

        # Since e is rewired, we have to update its subsets
        for size in range(2, len(e)):
            # For each subset, decrease the number of supersets by 1
            for subset in combinations(e, size):
                SuperSets[subset] -= 1
                # If the subset has no supersets and is not itself an edge,
                # rewiring the edge destroys the simplex
                if SuperSets[subset] == 0 and subset not in Edges:
                    Simplices.remove(subset)
                    SimplexCounts[timer][len(subset)-1] -= 1
                    Times[subset].append(timer)

        # Rewire edge
        source = np.random.choice(e)
        while True:
            newTargets = set([source])
            # Randomly sample new targets
            while len(newTargets) < len(e):
                newTarget = np.random.choice(nodes)
                newTargets.add(newTarget)
            newEdge = tuple(sorted(newTargets))
            if newEdge not in Edges:
                Edges.add(newEdge)
                H[timer + len(E) - 1] = newEdge
                break

        # Update new edge
        if newEdge not in Simplices:
            Simplices.add(newEdge)
            SimplexCounts[timer][len(newEdge)-1] += 1
            if newEdge not in Times:
                Times[newEdge] = [timer]
            else:
                Times[newEdge].append(timer)

        # We have a new edge so we have to update data structures for subsets
        for size in range(2, len(newEdge)):
            # For each subset, decrease the number of supersets by 1
            for subset in combinations(newEdge, size):
                # If the subset was not already a simplex, it is now
                if subset not in Simplices:
                    Simplices.add(subset)
                    SimplexCounts[timer][len(subset)-1] += 1
                    if subset not in Times:
                        Times[subset] = [timer]
                    else:
                        Times[subset].append(timer)
                # if the subset is a simplex AND is one of our initial edges, increment supersets
                elif subset in SuperSets:
                    SuperSets[subset] += 1

    if timing:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Zigzag Persistent Homology",flush=True)
        start = time.time()

    # Extract list of every simplex added/removed, and list of times they were
    # added/removed, for input into zigzag persistence.
    simplices = [list(key) for key in Times]; times = [Times[key] for key in Times]

    # Clear out Times, which is massive
    del(Times); del(Simplices); del(SuperSets); del(Edges); gc.collect()

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
        print("Betti number extraction complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(5/5) Beginning Euler Characteristic extraction",flush=True)
        start = time.time()

    # For each time step, compute the euler characteristic as the alternating
    # sum of simplex counts SUM( (-1)^j * num j-simplices )
    Euler = np.zeros(timer+1)
    for i in range(len(SimplexCounts)):
        for j in range(len(SimplexCounts[i])):
            Euler[i] += np.power(-1,j) * SimplexCounts[i][j]

    if timing:
        end = time.time()
        print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds",flush=True)

    return H, Betti, SimplexCounts, Euler

def HG_WattsStrogatz(n, K, timing = False):
    """
    ...
    Input: int n, the number of nodes in H
           int K, the maximum allowable edge size
           bool timing, whether to display timing information
    Output: Betti, an array of the 0, 1 and 2 Betti numbers for each time step
            SimplexCounts
    """
    if timing:
        print("(1/5) Initializing hypergraph, variables and data structures",flush=True)
        start = time.time()

    nodes = range(n)

    # Generate a list of all edges of size <= K that can be formed from n nodes
    E = [tuple(sorted([(v+i) % n for i in range(size)])) for v in nodes for size in range(2, K+1)]
    Edges = set()
    Edges.update(E)
    H = {i: e for i, e in enumerate(E)}
    # By shuffing this list, it is equivalent to forming a filtration where
    # edges are rewired u.a.r
    random.shuffle(E)

    # Use set structure to remember what simplices have been added
    Simplices = set(tuple([v]) for v in nodes)
    # Intialize simplex counts with n vertices
    # The maximum simplex dimension is either n-1 or maxDim
    SimplexCounts = np.zeros( (len(E)+1 , min(K, n)) )
    SimplexCounts[0][0] = n

    # Initialize dictionary which keeps track of when simplices
    # are added and removed
    Times = {tuple([v]) : [0] for v in nodes}
    timer = 0
    SuperSets = dict()

    # Need to keep track of how many supersets there are for the initial
    # simplices, as these are the only simplices that can be removed, which
    # only happens when 1. they are not an edge and 2. are not contained within
    # any other edges.
    for e in E:
        if e not in SuperSets:
            SuperSets[e] = 0
            Simplices.add(e)
            SimplexCounts[0][len(e)-1] += 1
            Times[e] = [0]
        for size in range(2, len(e)):
            for subset in combinations(e, size):
                if subset not in SuperSets:
                    SuperSets[subset] = 1
                    Simplices.add(subset)
                    SimplexCounts[0][len(subset)-1] += 1
                    Times[subset] = [0]
                else:
                    SuperSets[subset] += 1

    if timing:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    for e in tqdm(E) if timing else E:
        timer += 1; SimplexCounts[timer] = SimplexCounts[timer-1].copy();
        # e is being rewired so we remove it from edges
        H[-(timer + len(E) - 1)] = e
        Edges.remove(e)
        # If the removed edge has no supersets, then the simplex is destroyed
        if (e not in SuperSets) or (SuperSets[e] == 0):
            Simplices.remove(e)
            SimplexCounts[timer][len(e)-1] -= 1
            Times[e].append(timer)

        # Since e is rewired, we have to update its subsets
        for size in range(2, len(e)):
            # For each subset, decrease the number of supersets by 1
            for subset in combinations(e, size):
                SuperSets[subset] -= 1
                # If the subset has no supersets and is not itself an edge,
                # rewiring the edge destroys the simplex
                if SuperSets[subset] == 0 and subset not in Edges:
                    Simplices.remove(subset)
                    SimplexCounts[timer][len(subset)-1] -= 1
                    Times[subset].append(timer)

        # Rewire edge
        source = np.random.choice(e)
        while True:
            newTargets = set([source])
            # Randomly sample new targets
            while len(newTargets) < len(e):
                newTarget = np.random.choice(nodes)
                newTargets.add(newTarget)
            newEdge = tuple(sorted(newTargets))
            if newEdge not in Edges:
                Edges.add(newEdge)
                H[timer + len(E) - 1] = newEdge
                break

        # Update new edge
        if newEdge not in Simplices:
            Simplices.add(newEdge)
            SimplexCounts[timer][len(newEdge)-1] += 1
            if newEdge not in Times:
                Times[newEdge] = [timer]
            else:
                Times[newEdge].append(timer)

        # We have a new edge so we have to update data structures for subsets
        for size in range(2, len(newEdge)):
            # For each subset, decrease the number of supersets by 1
            for subset in combinations(newEdge, size):
                # If the subset was not already a simplex, it is now
                if subset not in Simplices:
                    Simplices.add(subset)
                    SimplexCounts[timer][len(subset)-1] += 1
                    if subset not in Times:
                        Times[subset] = [timer]
                    else:
                        Times[subset].append(timer)
                # if the subset is a simplex AND is one of our initial edges, increment supersets
                elif subset in SuperSets:
                    SuperSets[subset] += 1

    if timing:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Zigzag Persistent Homology",flush=True)
        start = time.time()

    # Extract list of every simplex added/removed, and list of times they were
    # added/removed, for input into zigzag persistence.
    simplices = [list(key) for key in Times]; times = [Times[key] for key in Times]

    # Clear out Times, which is massive
    del(Times); del(Simplices); del(SuperSets); del(Edges); gc.collect()

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
        print("Betti number extraction complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(5/5) Beginning Euler Characteristic extraction",flush=True)
        start = time.time()

    # For each time step, compute the euler characteristic as the alternating
    # sum of simplex counts SUM( (-1)^j * num j-simplices )
    Euler = np.zeros(timer+1)
    for i in range(len(SimplexCounts)):
        for j in range(len(SimplexCounts[i])):
            Euler[i] += np.power(-1,j) * SimplexCounts[i][j]

    if timing:
        end = time.time()
        print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds",flush=True)

    return H, Betti, SimplexCounts, Euler

def HG_WattsStrogatz_Simplicial(n, K, p, timing = False):
    """
    ...
    Input: int n, the number of nodes in H
           int K, the maximum allowable edge size
           bool timing, whether to display timing information
    Output: Betti, an array of the 0, 1 and 2 Betti numbers for each time step
            SimplexCounts
    """
    if timing:
        print("(1/5) Initializing hypergraph, variables and data structures",flush=True)
        start = time.time()

    nodes = range(n)

    # Generate a list of all edges of size <= K that can be formed from n nodes
    E = [tuple(sorted([(v+i) % n for i in range(K)])) for v in nodes]
    for v in nodes:
        for size in range(2,K):
            for comb in combinations(range(v+1,v+K),size-1):
                if random.random() < p:
                    E.append(tuple(sorted([v]+[u % n for u in comb])))

    Edges = set()
    Edges.update(E)
    H = {i: e for i, e in enumerate(E)}
    # By shuffing this list, it is equivalent to forming a filtration where
    # edges are rewired u.a.r
    random.shuffle(E)

    # Use set structure to remember what simplices have been added
    Simplices = set(tuple([v]) for v in nodes)
    # Intialize simplex counts with n vertices
    # The maximum simplex dimension is either n-1 or maxDim
    SimplexCounts = np.zeros( (len(E)+1 , min(K, n)) )
    SimplexCounts[0][0] = n

    # Initialize dictionary which keeps track of when simplices
    # are added and removed
    Times = {tuple([v]) : [0] for v in nodes}
    timer = 0
    SuperSets = dict()

    # Need to keep track of how many supersets there are for the initial
    # simplices, as these are the only simplices that can be removed, which
    # only happens when 1. they are not an edge and 2. are not contained within
    # any other edges.
    for e in E:
        if e not in SuperSets:
            SuperSets[e] = 0
            Simplices.add(e)
            SimplexCounts[0][len(e)-1] += 1
            if len(e) <= 4:
                Times[e] = [0]
        for size in range(2, len(e)):
            for subset in combinations(e, size):
                if subset not in SuperSets:
                    SuperSets[subset] = 1
                    Simplices.add(subset)
                    SimplexCounts[0][len(subset)-1] += 1
                    if len(subset) <= 4:
                        Times[subset] = [0]
                else:
                    SuperSets[subset] += 1

    if timing:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    for e in tqdm(E) if timing else E:
        timer += 1; SimplexCounts[timer] = SimplexCounts[timer-1].copy();
        # e is being rewired so we remove it from edges
        H[-(timer + len(E) - 1)] = e
        Edges.remove(e)
        # If the removed edge has no supersets, then the simplex is destroyed
        if (e not in SuperSets) or (SuperSets[e] == 0):
            Simplices.remove(e)
            SimplexCounts[timer][len(e)-1] -= 1
            if len(e) <= 4:
                Times[e].append(timer)

        # Since e is rewired, we have to update its subsets
        for size in range(2, len(e)):
            # For each subset, decrease the number of supersets by 1
            for subset in combinations(e, size):
                SuperSets[subset] -= 1
                # If the subset has no supersets and is not itself an edge,
                # rewiring the edge destroys the simplex
                if SuperSets[subset] == 0 and subset not in Edges:
                    Simplices.remove(subset)
                    SimplexCounts[timer][len(subset)-1] -= 1
                    if len(subset) <= 4:
                        Times[subset].append(timer)

        # Rewire edge
        source = np.random.choice(e)
        while True:
            newTargets = set([source])
            # Randomly sample new targets
            while len(newTargets) < len(e):
                newTarget = np.random.choice(nodes)
                newTargets.add(newTarget)
            newEdge = tuple(sorted(newTargets))
            if newEdge not in Edges:
                Edges.add(newEdge)
                H[timer + len(E) - 1] = newEdge
                break

        # Update new edge
        if newEdge not in Simplices:
            Simplices.add(newEdge)
            SimplexCounts[timer][len(newEdge)-1] += 1
            if len(newEdge) <= 4:
                if newEdge not in Times:
                    Times[newEdge] = [timer]
                else:
                    Times[newEdge].append(timer)

        # We have a new edge so we have to update data structures for subsets
        for size in range(2, len(newEdge)):
            # For each subset, decrease the number of supersets by 1
            for subset in combinations(newEdge, size):
                # If the subset was not already a simplex, it is now
                if subset not in Simplices:
                    Simplices.add(subset)
                    SimplexCounts[timer][len(subset)-1] += 1
                    if len(subset) <= 4:
                        if subset not in Times:
                            Times[subset] = [timer]
                        else:
                            Times[subset].append(timer)
                # if the subset is a simplex AND is one of our initial edges, increment supersets
                elif subset in SuperSets:
                    SuperSets[subset] += 1

    if timing:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Zigzag Persistent Homology",flush=True)
        start = time.time()

    # Extract list of every simplex added/removed, and list of times they were
    # added/removed, for input into zigzag persistence.
    simplices = [list(key) for key in Times]; times = [Times[key] for key in Times]

    # Clear out Times, which is massive
    del(Times); del(Simplices); del(SuperSets); del(Edges); gc.collect()

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
        print("Betti number extraction complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(5/5) Beginning Euler Characteristic extraction",flush=True)
        start = time.time()

    # For each time step, compute the euler characteristic as the alternating
    # sum of simplex counts SUM( (-1)^j * num j-simplices )
    Euler = np.zeros(timer+1)
    for i in range(len(SimplexCounts)):
        for j in range(len(SimplexCounts[i])):
            Euler[i] += np.power(-1,j) * SimplexCounts[i][j]

    if timing:
        end = time.time()
        print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds",flush=True)

    return H, Betti, SimplexCounts, Euler

def PA_Poisson_Binomial_HypergraphModel(lam, p, steps, timing=False): # P_x = 'default',
  """
  Input: 
  Output: Hypergraph dictionary H along with the degree frequency distribution D.
  """

  # Initialize node list V, and hypergraph as a dictionary will lam many
  # nodes belonging to a hyperedge of size lam
  V = list(range(lam))
  H = {0:[i for i in range(lam)]}

  # Keep track of vertex degrees for degree distribution
  D = [1] * lam
  # We will use repeated_nodes as a fast, but storage intensive way to
  # preferentially draw nodes. Each node v with appear deg(v) many times.
  repeated_nodes = [i for i in range(lam)];

  for step in (tqdm(range(1,steps+1)) if timing else range(1,steps+1)):
    # Select the size of Yt from the Poisson distribution
    Y_t = np.random.poisson(lam)
    # Select the number of nodes Xt from a binomial distribution with
    # E[Xt | Yt] = p*Yt
    X_t = np.random.binomial(Y_t, p)
    targets = set()

    # Check to make sure there are enough nodes to wire to
    # If not, add more new nodes instead of pref wiring
    if (Y_t - X_t) > len(V):
        X_t += (Y_t - X_t) - len(V)
    # Select (Y_t - X_t) many targets in graph using preferential attachment
    while len(targets) < (Y_t - X_t):
        x = random.choice(repeated_nodes)
        targets.add(x)
    targets = list(targets)
    # Add the X_t many new nodes to the new hyperedge
    new_nodes = [V[-1] + 1 + i for i in range(X_t)]
    V.extend(new_nodes); D.extend([0] * X_t)
    H_edge = targets + new_nodes; H[step] = H_edge
    repeated_nodes.extend(H_edge)

    # Update node degrees
    for v in H_edge: D[v] += 1

  # Returns D as a dictionary with degree values as keys and
  # the number of nodes with the given degree as entries.
  D =  collections.Counter(D)

  return H, D

def PA_Dist_From_Data_HypergraphModel(Xt, Yt, timing=False):
  """
  Input:
  Output: Hypergraph dictionary H along with the degree frequency distribution D.
  """

  # Initialize node list V, and hypergraph as a dictionary will max(Yt) many
  # nodes belonging to a hyperedge of size max(Yt)
  V = list(range(Yt[0]))
  H = {0:[i for i in range(Yt[0])]}

  # Keep track of vertex degrees for degree distribution
  D = [1] * Yt[0]
  # We will use repeated_nodes as a fast, but storage intensive way to
  # preferentially draw nodes. Each node v with appear deg(v) many times.
  repeated_nodes = [i for i in range(Yt[0])];

  for step in (tqdm(range(1,len(Yt))) if timing else range(1,len(Yt))):
    # Select the size of Yt
    yt = Yt[step]
    # Select the number of nodes Xt
    xt = Xt[step]
    targets = set()

    # Select (Y_t - X_t) many targets in graph using preferential attachment
    while len(targets) < (yt - xt):
        x = random.choice(repeated_nodes)
        targets.add(x)
    targets = list(targets)
    # Add the X_t many new nodes to the new hyperedge
    new_nodes = [V[-1] + 1 + i for i in range(xt)]
    V.extend(new_nodes); D.extend([0] * xt)
    H_edge = targets + new_nodes; H[step] = H_edge
    repeated_nodes.extend(H_edge)

    # Update node degrees
    for v in H_edge: D[v] += 1

  # Returns D as a dictionary with degree values as keys and
  # the number of nodes with the given degree as entries.
  D =  collections.Counter(D)

  return H, D

def RA_Dist_From_Data_HypergraphModel(Xt, Yt, timing=False):
  """
  Input:
  Output: Hypergraph dictionary H along with the degree frequency distribution D.
  """

  # Initialize node list V, and hypergraph as a dictionary
  V = list(range(Yt[0]))
  H = {0:[i for i in range(Yt[0])]}

  # Keep track of vertex degrees for degree distribution
  D = [1] * Yt[0]

  for step in (tqdm(range(1,len(Yt))) if timing else range(1,len(Yt))):
    # Select the size of Yt
    yt = Yt[step]
    # Select the number of nodes Xt
    xt = Xt[step]
    targets = set()

    # Select (Y_t - X_t) many targets in graph using preferential attachment
    while len(targets) < (yt - xt):
        x = random.choice(V)
        targets.add(x)
    targets = list(targets)
    # Add the X_t many new nodes to the new hyperedge
    new_nodes = [V[-1] + 1 + i for i in range(xt)]
    V.extend(new_nodes); D.extend([0] * xt)
    H_edge = targets + new_nodes; H[step] = H_edge

    # Update node degrees
    for v in H_edge: D[v] += 1

  # Returns D as a dictionary with degree values as keys and
  # the number of nodes with the given degree as entries.
  D =  collections.Counter(D)

  return H, D

def Nonlinear_PA_Dist_From_Data_HypergraphModel(Xt, Yt, alpha=1.0, timing=False):
    """
    Input:
         Xt - list of number of new nodes to add at each step
         Yt - list of hyperedge sizes at each step
         alpha - nonlinear preferential attachment parameter
         timing - boolean to show progress bar
    Output: Hypergraph dictionary H along with the degree frequency distribution D.
    """

    num_nodes = Yt[0]
    H = {0: [i for i in range(Yt[0])]}

    # Keep track of vertex degrees for degree distribution
    D = [1] * Yt[0]

    # Initialize efficient weighted sampler
    initial_weights = [1.0] * Yt[0]
    sampler = WeightedSampler(initial_weights)

    total_nodes = sum(Xt)
    D.extend([0] * (total_nodes - Yt[0]))

    for step in (tqdm(range(1, len(Yt))) if timing else range(1, len(Yt))):
        # Select the size of Yt and number of nodes Xt
        yt = Yt[step]
        xt = Xt[step]

        targets_needed = yt - xt
        targets = set()

        while len(targets) < targets_needed:
            x = sampler.sample(exclude=targets)
            if x is not None:
                targets.add(x)
            else:
                break

        targets = list(targets)

        # Add the X_t many new nodes to the new hyperedge
        new_nodes = [num_nodes + i for i in range(xt)]
        num_nodes += xt
        H_edge = targets + new_nodes
        H[step] = H_edge

        # Batch update degrees and weights
        for v in H_edge:
            D[v] += 1
            new_degree = D[v]
            new_weight = new_degree ** alpha
            sampler.update_weight(v, new_weight)

    D = collections.Counter(D)

    return H, D

def RandomShuffling(filename):
  if type(filename) == dict:
    H = filename
  else:
    with gzip.open(filename, 'rb') as f:
      try:
        with gzip.open(filename, 'rb') as f:
          H, _ = pickle.load(f)
      except Exception as e:
        print(f"Error loading file: {filename}")
        return -1

  V = set()
  for e in H:
    for v in H[e]:
      V.add(v)

  V = list(V)
  for e in H:
    H[e] = random.sample(V,len(H[e]))

  return H

def ProportionalShuffling(filename):
  if type(filename) == dict:
    H = filename
  else:
    with gzip.open(filename, 'rb') as f:
      try:
        with gzip.open(filename, 'rb') as f:
          H, _ = pickle.load(f)
      except Exception as e:
        print(f"Error loading file: {filename}")
        return -1

  repeated_nodes = [v for e in H for v in H[e]]

  for e in H:
    targets = set()
    while len(targets) < len(H[e]):
        x = random.choice(repeated_nodes)
        targets.add(x)
    targets = list(targets)
    H[e] = targets

  return H

def TemporalShuffling(filename):
  if type(filename) == dict:
    H = filename
  else:
    with gzip.open(filename, 'rb') as f:
      try:
        with gzip.open(filename, 'rb') as f:
          H, _ = pickle.load(f)
      except Exception as e:
        print(f"Error loading file: {filename}")
        return -1

  keys = list(H.keys())
  random.shuffle(keys)
  H_shuffled = {i: H[keys[i]] for i in range(len(keys))}

  return H_shuffled

def HyperdegreePreservingShuffling(filename, numSwaps):
  if type(filename) == dict:
    H = filename
  else:
    with gzip.open(filename, 'rb') as f:
      try:
        with gzip.open(filename, 'rb') as f:
          H, _ = pickle.load(f)
      except Exception as e:
        print(f"Error loading file: {filename}")
        return -1

  layers = {}; keys = list(H.keys())
  for e in H:
    if len(H[e]) in layers:
      layers[len(H[e])].append(e)
    else:
      layers[len(H[e])] = [e]

  for swap in range(numSwaps):
    while True:
      e1 = random.choice(keys)
      if len(H[e1]) > 0 and len(layers[len(H[e1])]) > 1:
        break
    while True:
      e2 = random.choice(layers[len(H[e1])])
      if e2 != e1:
        break
    i1 = random.randint(0,len(H[e1])-1)
    i2 = random.randint(0,len(H[e2])-1)
    tempV = H[e1][i1]
    H[e1][i1] = H[e2][i2]
    H[e2][i2] = tempV

  return H

def LayerPreservingShuffling(filename):
  if type(filename) == dict:
    H = filename
  else:
    with gzip.open(filename, 'rb') as f:
      try:
        with gzip.open(filename, 'rb') as f:
          H, _ = pickle.load(f)
      except Exception as e:
        print(f"Error loading file: {filename}")
        return -1

  V = set()
  layers = {}; keys = list(H.keys())
  for e in H:
    for v in H[e]:
      V.add(v)

    if len(H[e]) in layers:
      layers[len(H[e])].append(e)
    else:
      layers[len(H[e])] = [e]

  V = sorted(list(V))
  for layer in layers:
    V_shuffled = V.copy()
    random.shuffle(V_shuffled)
    for e in layers[layer]:
      H[e] = [V_shuffled[V.index(v)] for v in H[e]]

  return H

# ============================================================================
# Parallel Calls to Models
# ============================================================================

def PC_HG_ErdosRenyi(params):
  """
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
  """
  # Grab parameter values from params list
  n = params[0]; k = params[1]; p = params[2]; iteration = params[3]; model = params[4]

  # Create filename from params
  if model == 'kunif':
    filename = 'Hypergraphs/ErdosRenyi/kunif_ER_'+str(n)+'_'+str(k)+'_'+str(p).replace('.','_')+'_'+str(iteration)+'.pkl'
    if os.path.isfile(filename):
      return 0
    H, Betti, SimplexCounts, Euler = HG_ErdosRenyi_kUnif(n, k, p, timing = False)
    Upper, Lower, SF, FES = simpliciality_process_hypergraph(H)
    Upper_sum = np.cumsum(Upper)
    Lower_sum = np.cumsum(Lower)
    data = [Betti, SimplexCounts, Euler, Upper_sum, Lower_sum, SF, FES]
  elif model == 'regular':
    filename = 'Hypergraphs/ErdosRenyi/ER_'+str(n)+'_'+str(k)+'_'+str(p).replace('.','_')+'_'+str(iteration)+'.pkl'
    if os.path.isfile(filename):
      return 0
    H, Betti, SimplexCounts, EdgeCounts, Euler = HG_ErdosRenyi(n, k, p, timing = False)
    Upper, Lower, SF, FES = simpliciality_process_hypergraph(H)
    Upper_sum = np.cumsum(Upper)
    Lower_sum = np.cumsum(Lower)
    data = [Betti, SimplexCounts, EdgeCounts, Euler, Upper_sum, Lower_sum, SF, FES]
  else:
    print("Invalid model")
    return 0

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def PC_HG_WattsStrogatz(params):
  """
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
  """
  # Grab parameter values from params list
  n = params[0]; k = params[1]; iteration = params[2]; model = params[3]

  # Create filename from params
  if model == 'kunif':
    filename = 'Hypergraphs/WattsStrogatz/kunif_WS_'+str(n)+'_'+str(k)+'_'+str(iteration)+'.pkl'
    if os.path.isfile(filename):
      return 0
    H, Betti, SimplexCounts, Euler = HG_WattsStrogatz_kUnif(n, k, timing = False)
    data = [Betti, SimplexCounts, Euler]
  elif model == 'regular':
    filename = 'Hypergraphs/WattsStrogatz/WS_'+str(n)+'_'+str(k)+'_'+str(iteration)+'.pkl'
    if os.path.isfile(filename):
      return 0
    H, Betti, SimplexCounts, Euler = HG_WattsStrogatz(n, k, timing = False)
    Upper, Lower, SF, FES = simpliciality_process_hypergraph(H)
    Upper_sum = np.cumsum(Upper)
    Lower_sum = np.cumsum(Lower)
    data = [Betti, SimplexCounts, Euler, Upper_sum, Lower_sum, SF, FES]
  else:
    print("Invalid model")
    return 0

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def PC_HG_Simplicial_WattsStrogatz(params):
  """
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
  """
  # Grab parameter values from params list
  n = params[0]; k = params[1]; p = params[2]; iteration = params[3];

  # Create filename from params
  filename = 'Hypergraphs/WattsStrogatz/WS_Simplicial_'+str(n)+'_'+str(k)+'_'+str(p).replace('.','_')+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
      return 0
  H, Betti, SimplexCounts, Euler = HG_WattsStrogatz_Simplicial(n, k, p, timing = False)
  Upper, Lower, SF, FES = simpliciality_process_hypergraph(H)
  Upper_sum = np.cumsum(Upper)
  Lower_sum = np.cumsum(Lower)
  data = [Betti, SimplexCounts, Euler, Upper_sum, Lower_sum, SF[len(H)//3:], FES[len(H)//3:]]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def PC_HG_BarabasiAlbert(params):
  """
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
  """
  # Grab parameter values from params list
  n = params[0]; k = params[1]; p = params[2]; iteration = params[3];

  # Create filename from params
  if p == 0:
    filename = 'Hypergraphs/BarabasiAlbert/BA_'+str(n)+'_'+str(k)+'_'+str(iteration)+'.pkl'
  else:
    filename = 'Hypergraphs/BarabasiAlbert/Simplicial_BA_'+str(n)+'_'+str(k)+'_'+str(p).replace('.','_')+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
    return 0

  H, D, Betti, SimplexCounts, Euler = HG_PreferentialAttachment_Simplicial(k, p, n, timing = False)
  Upper, Lower, SF, FES = simpliciality_process_hypergraph(H)
  Upper_sum = np.cumsum(Upper)
  Lower_sum = np.cumsum(Lower)
  data = [D, Betti, SimplexCounts, Euler, Upper_sum, Lower_sum, SF, FES]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def PC_HG_NL_BarabasiAlbert(params):
  """
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
  """
  # Grab parameter values from params list
  n = params[0]; k = params[1]; alpha = params[2]; iteration = params[3];
  print(n)
  # Create filename from params
  if alpha == 1:
    filename = 'Hypergraphs/BarabasiAlbert/BA_'+str(n)+'_'+str(k)+'_'+str(iteration)+'.pkl'
  elif alpha == 0:
    filename = 'Hypergraphs/BarabasiAlbert/RA_'+str(n)+'_'+str(k)+'_'+str(iteration)+'.pkl'
  else:
    filename = 'Hypergraphs/BarabasiAlbert/Nonlinear_BA_'+str(n)+'_'+str(k)+'_'+str(alpha).replace('.','_')+'_'+str(iteration)+'.pkl'
  if os.path.isfile(filename):
    return 0

  H, D, Betti, SimplexCounts, Euler = HG_NL_PreferentialAttachment_kUnif(k, n, alpha, timing = False)
  data = [D, Betti, SimplexCounts, Euler]

  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0

def PC_PA_Poisson_Binomial_HypergraphModel(params):
  """
  Parallel function which is iteratively called,
  looping over a set of parameters, with the current
  iteration of parameters being params.
  """
  # Grab parameter values from params list
  lam = params[0]; p = params[1]; N = params[2]; iteration = params[3]
  # By setting the number of steps in this way, we ensure that the
  # EXPECTED number of nodes at termination is at least N + N0.
  steps = math.ceil(N / (p * lam))
  # Create filename from params
  filename = 'Hypergraphs/PA_Poisson_Binomial_HypergraphModel/PA_Poisson_Binomial_HypergraphModel_'+str(lam)+'_'+str(p).replace('.','_')+'_'+str(N)+'_'+str(iteration)+'.pkl'
  # Check if current file already exists, if it does exist we do not
  # want to waste time running it again
  if os.path.isfile(filename):
    return 0
  data = PA_Poisson_Binomial_HypergraphModel(lam, p, steps, timing = True)
  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  return 0
  
def PC_Simplices_Euler(params):
  lam, p, N, iteration = params

  # Check whether this data has been processed already
  processed_filename = 'Hypergraphs/Processed/Simplices_Euler/PA_Poisson_Binomial_HypergraphModel_Simplices_Euler_'+str(lam)+'_'+str(p).replace('.','_')+'_'+str(N)+'_'+str(iteration)+'.pkl'
  if os.path.isfile(processed_filename):
    return 1

  # For each run, load network, extract degree sequence and appending to total degree sequence
  filename = 'Hypergraphs/PA_Poisson_Binomial_HypergraphModel/PA_Poisson_Binomial_HypergraphModel_'+str(lam)+'_'+str(p).replace('.','_')+'_'+str(N)+'_'+str(iteration)+'.pkl'
  results = Get_SimplexCounts_EulerChar(filename, processed_filename, lam, maxDim = np.inf)

  return results

def PC_SimplicialFraction(params):
  lam, p, N, iteration = params

  # Check whether this data has been processed already
  processed_filename = 'Hypergraphs/Processed/Simpliciality/SimplicialFraction/PA_Poisson_Binomial_HypergraphModel_SF_'+str(lam)+'_'+str(p).replace('.','_')+'_'+str(N)+'_'+str(iteration)+'.pkl'
  if os.path.isfile(processed_filename):
    return 1

  # For each run, load network, extract degree sequence and appending to total degree sequence
  filename = 'Hypergraphs/PA_Poisson_Binomial_HypergraphModel/PA_Poisson_Binomial_HypergraphModel_'+str(lam)+'_'+str(p).replace('.','_')+'_'+str(N)+'_'+str(iteration)+'.pkl'
  results = Get_Simpliciality_SF(filename, processed_filename, minDim = 2, maxDim = np.inf)

  return results

def PC_PA_From_Data(params):
  name, step = params

  filename = 'Hypergraphs/PA_From_Data/'+str(name)+'/PA_'+str(name)+'_'+str(step)+'.pkl'
  if os.path.exists(filename):
    print(f"File {filename} already exists. Skipping...")
    return 1

  loaddata = 'Hypergraphs/Processed/XtYt_FromData/'+str(name)+'.pkl'
  with gzip.open(loaddata, 'rb') as f:
    extracted_info = pickle.load(f)
  # Xt and Yt's are recorded in reverse order so we reverse them back
  extracted_info.reverse()
  Xt = []; Yt = []
  for entry in extracted_info:
    Yt.append(entry['edge_size'])
    Xt.append(entry['isolated_nodes'])

  data = PA_Dist_From_Data_HypergraphModel(Xt, Yt, timing=False)
  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  print(f"Saved {filename}")

  return 0

def PC_RA_From_Data(params):
  name, step = params

  filename = 'Hypergraphs/RA_From_Data/'+str(name)+'/RA_'+str(name)+'_'+str(step)+'.pkl'
  if os.path.exists(filename):
    print(f"File {filename} already exists. Skipping...")
    return 1

  loaddata = 'Hypergraphs/Processed/XtYt_FromData/'+str(name)+'.pkl'
  with gzip.open(loaddata, 'rb') as f:
    extracted_info = pickle.load(f)
  # Xt and Yt's are recorded in reverse order so we reverse them back
  extracted_info.reverse()
  Xt = []; Yt = []
  for entry in extracted_info:
    Yt.append(entry['edge_size'])
    Xt.append(entry['isolated_nodes'])

  data = RA_Dist_From_Data_HypergraphModel(Xt, Yt, timing=False)
  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  print(f"Saved {filename}")

  return 0

def PC_Nonlinear_PA_From_Data(params):
  name, alpha, step = params

  filename = 'Hypergraphs/PA_From_Data/'+str(name)+'/Nonlinear_'+f"{alpha:.1f}".replace('.','_')+'_'+str(name)+'_'+str(step)+'.pkl'
  if os.path.exists(filename):
    print(f"File {filename} already exists. Skipping...")
    return 1

  loaddata = 'Hypergraphs/Processed/XtYt_FromData/'+str(name)+'.pkl'
  with gzip.open(loaddata, 'rb') as f:
    extracted_info = pickle.load(f)
  # Xt and Yt's are recorded in reverse order so we reverse them back
  extracted_info.reverse()
  Xt = []; Yt = []
  for entry in extracted_info:
    Yt.append(entry['edge_size'])
    Xt.append(entry['isolated_nodes'])

  data = Nonlinear_PA_Dist_From_Data_HypergraphModel(Xt, Yt, alpha, timing=False)
  # Pickle and save the output
  with gzip.open(filename,'wb') as f:
    pickle.dump(data, f);
  print(f"Saved {filename}")

  return 

# ============================================================================
# Loading Data
# ============================================================================

def get_filename_ER(n, k, p, iteration, model, base_path):
    if model == 'kunif':
        return os.path.join(
            base_path,
            f'kunif_ER_{n}_{k}_{str(p).replace(".","_")}_{iteration}.pkl'
        )
    elif model == 'nonuniform':
        return os.path.join(
            base_path,
            f'ER_{n}_{k}_{str(p).replace(".","_")}_{iteration}.pkl'
        )

def load_single_er(filename):
    if not os.path.isfile(filename):
        return None
    try:
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f'Corrupted file skipped ({type(e).__name__}): {filename}')
        return None

def get_filename_ws(n, k, p, iteration, base_path):
    return os.path.join(
        base_path,
        f'WS_Simplicial_{n}_{k}_{str(p).replace(".","_")}_{iteration}.pkl'
    )

def load_single_ws(n, k, p, iteration, base_path, WS_fields):
    """
    Load one file and return all derived quantities.
    Returns (n, k, p_key, iteration, field_dict, status).
    """
    p_key    = round(p, 4)
    filename = get_filename_ws(n, k, p, iteration, base_path)

    if not os.path.isfile(filename):
        return n, k, p_key, iteration, None, 'missing'
    try:
        with gzip.open(filename, 'rb') as f:
            result = pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        return n, k, p_key, iteration, None, f'corrupted: {type(e).__name__}'

    field_dict = {field: value for field, value in zip(WS_fields, result)}

    betti          = result[WS_fields.index('Betti')]
    simplex_counts = result[WS_fields.index('SimplexCounts')]

    if betti is not None and simplex_counts is not None:
        field_dict['CycleRank']              = cyclerank_1(betti, simplex_counts)
        field_dict['CycleRank_minus_Betti1'] = (field_dict['CycleRank']
                                                 - betti[1].astype(float))
    else:
        field_dict['CycleRank']              = None
        field_dict['CycleRank_minus_Betti1'] = None

    return n, k, p_key, iteration, field_dict, 'ok'

def get_filename_nlba(n, k, alpha, iteration, base_path):
    if alpha == 1:
        return os.path.join(base_path, f'BA_{n}_{k}_{iteration}.pkl')
    elif alpha == 0:
        return os.path.join(base_path, f'RA_{n}_{k}_{iteration}.pkl')
    else:
        return os.path.join(
            base_path,
            f'Nonlinear_BA_{n}_{k}_{str(alpha).replace(".","_")}_{iteration}.pkl'
        )

def load_single_nlba(n, k, alpha, iteration, base_path, NLBA_fields):
    """
    Load one file and return all derived quantities.
    Returns (n, k, alpha_key, iteration, field_dict, status).
    """
    alpha_key = round(alpha, 4)
    filename  = get_filename_nlba(n, k, alpha, iteration, base_path)

    if not os.path.isfile(filename):
        return n, k, alpha_key, iteration, None, 'missing'
    try:
        with gzip.open(filename, 'rb') as f:
            result = pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        return n, k, alpha_key, iteration, None, f'corrupted: {type(e).__name__}'

    # Build field dict for this single iteration
    field_dict = {field: value for field, value in zip(NLBA_fields, result)}

    betti          = result[NLBA_fields.index('Betti')]
    simplex_counts = result[NLBA_fields.index('SimplexCounts')]

    if betti is not None and simplex_counts is not None:
        field_dict['CycleRank']             = cyclerank_1(betti, simplex_counts)
        field_dict['CycleRank_minus_Betti1'] = (field_dict['CycleRank']
                                                 - betti[1].astype(float))
    else:
        field_dict['CycleRank']             = None
        field_dict['CycleRank_minus_Betti1'] = None

    return n, k, alpha_key, iteration, field_dict, 'ok'

def sc_col(sc, dim):
    return np.array([
        row[dim] if len(row) > dim else 0.0
        for row in sc
    ], dtype=float)

def cyclerank_1(betti, sc):
    return sc_col(sc, 1) - sc_col(sc, 0) + betti[0].astype(float)

def cyclerank_2(betti, sc):
    return sc_col(sc, 2) - cyclerank_1(betti, sc) + betti[1].astype(float)

def cR2_terminal(betti, sc):
    N2  = float(sc[-1, 2])
    CR1 = float(sc[-1, 1] - sc[-1, 0] + betti[0][-1])
    b1  = float(betti[1][-1])
    return N2 - CR1 + b1

def tb1(betti, sc):
    b1 = betti[1].astype(float)
    s1 = sc_col(sc, 1)
    L  = min(len(b1), len(s1))
    return np.where(s1[:L] > 0, b1[:L] / s1[:L], np.nan)

def tb2(betti, sc):
    b2 = betti[2].astype(float)
    s2 = sc_col(sc, 2)
    L  = min(len(b2), len(s2))
    return np.where(s2[:L] > 0, b2[:L] / s2[:L], np.nan)

def build_heatmap(data, n_plot, n_q, k, q_common, field_fn, p_keys):
    rows = []
    for p in p_keys:
        rows.append(mean_term_traj(data, n_plot, n_q, k, q_common, field_fn, p))
    return np.array(rows)   # (n_p, n_q)

def extract_CCDF(N, lam, p, iterations):
    # Check whether this data has been processed already
    processed_filename = 'Hypergraphs/Processed/CCDF/PA_Poisson_Binomial_HypergraphModel_CCDF_'+str(lam)+'_'+str(p).replace('.','_')+'_'+str(N)+'.pkl'
    if os.path.isfile(processed_filename):
      return 0

    total_degree = {}
    for iteration in tqdm(iterations):
    # For each run, load network, extract degree sequence and appending to total degree sequence
      filename = 'Hypergraphs/PA_Poisson_Binomial_HypergraphModel/PA_Poisson_Binomial_HypergraphModel_'+str(lam)+'_'+str(p).replace('.','_')+'_'+str(N)+'_'+str(iteration)+'.pkl'
      try:
        with gzip.open(filename, 'rb') as f:
            _, degree = pickle.load(f)
        for key in degree:
          if key in total_degree:
            total_degree[key] += degree[key]
          else:
            total_degree[key] = degree[key]

      except Exception as e:
                print(f"Error loading file: {filename}")

    # Turn degree sequence into dictionary containing (node degree: # of nodes with given degree) key-value pairs
    maxDegree = max(total_degree.keys())
    P_PB_HG_c = np.array([0] * (maxDegree+1))
    P_PB_HG_c[maxDegree-1] = total_degree[maxDegree]

    # Obtain CCDF
    for i in range(maxDegree-2,-1,-1):
      P_PB_HG_c[i] = P_PB_HG_c[i+1]
      if (i+1) in total_degree:
        P_PB_HG_c[i] += total_degree[i+1]

    P_PB_HG_c = P_PB_HG_c/P_PB_HG_c[0]

    # Pickle and save the output
    with gzip.open(processed_filename, 'wb') as f:
      pickle.dump(P_PB_HG_c, f)

def load_ccdf(N, lam, p):
    filename = (
        'Hypergraphs/Processed/CCDF/'
        f'PA_Poisson_Binomial_HypergraphModel_CCDF_{lam}_{str(p).replace(".","_")}_{N}.pkl'
    )
    try:
        with gzip.open(filename, 'rb') as f:
            ccdf = pickle.load(f)
        return (lam, p), ccdf, None
    except Exception as e:
        return (lam, p), None, str(e)

def load_euler(N, lam, p, iteration):
    filename = (
        'Hypergraphs/Processed/Simplices_Euler/'
        f'PA_Poisson_Binomial_HypergraphModel_Simplices_Euler_{lam}_{str(p).replace(".","_")}_{N}_{iteration}.pkl'
    )
    try:
        with gzip.open(filename, 'rb') as f:
            _, euler = pickle.load(f)
        return (lam, p, iteration), euler, None
    except Exception as e:
        return (lam, p, iteration), None, str(e)

def load_sf(N, lam, p, iteration):
    filename = (
        'Hypergraphs/Processed/Simpliciality/SimplicialFraction/'
        f'PA_Poisson_Binomial_HypergraphModel_SF_{lam}_{str(p).replace(".","_")}_{N}_{iteration}.pkl'
    )
    try:
        with gzip.open(filename, 'rb') as f:
            sf, lengths = pickle.load(f)
        return (lam, p, iteration), sf, lengths, None
    except Exception as e:
        return (lam, p, iteration), None, None, str(e)

def Convert_From_XGI(name):
  H = xgi.load_xgi_data(name)
  counter = 0
  H_processed = {}
  for e in H.edges:
    H_processed[counter] = tuple(sorted(H.edges.members(e)))
    counter += 1
  return H_processed

def load_original(name, base_path):
    filename = base_path + f'{name}.pkl'
    try:
        with gzip.open(filename, 'rb') as f:
            return name, pickle.load(f)[0], None
    except Exception as e:
        return name, None, str(e)

def load_fromdata(name, step, base_path):
    filename = base_path + f'{name}_{step}.pkl'
    try:
        with gzip.open(filename, 'rb') as f:
            return name, step, pickle.load(f)[0], None
    except Exception as e:
        return name, step, None, str(e)

def load_nonlinear(name, alpha, step, base_path):
    filename = base_path + f'Nonlinear_{str(alpha).replace(".","_")}_{name}_{step}.pkl'
    try:
        with gzip.open(filename, 'rb') as f:
            return name, alpha, step, pickle.load(f)[0], None
    except Exception as e:
        return name, alpha, step, None, str(e)

def load_ra_and_shuffled(name, step, base_path):
    ra_filename  = base_path + f'RA_{name}_{step}.pkl'
    shuf_filename = base_path + f'Shuffled_{name}_{step}.pkl'
    try:
        with gzip.open(ra_filename, 'rb') as f:
            ra = pickle.load(f)[0]
        with gzip.open(shuf_filename, 'rb') as f:
            shuf = pickle.load(f)
        return (name, step, ra,
                shuf[0][0], shuf[1][0], shuf[2][0], shuf[3][0], None)
    except Exception as e:
        return name, step, None, None, None, None, None, str(e)

# ============================================================================
# Plotting
# ============================================================================

def add_discrete_cbar_n(fig, ax_target, n_cmap, n_norm, n_n, N, shrink=0.6):
    sm = cm.ScalarMappable(cmap=n_cmap, norm=n_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_target, shrink=shrink, pad=0.02,
                        location='right', ticks=np.arange(n_n))
    cbar.set_ticklabels([str(n) for n in N])
    cbar.set_label(r'$N$ (number of nodes)', fontsize=12)
    return cbar

def add_discrete_cbar_k(fig, ax_target, k_cmap, k_norm, n_k, K, shrink=0.6):
    sm = cm.ScalarMappable(cmap=k_cmap, norm=k_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_target, shrink=shrink, pad=0.02,
                        location='right', ticks=np.arange(n_k))
    cbar.set_ticklabels([str(k) for k in K])
    cbar.set_label(r'$K$ (hyperedge size)', fontsize=13)
    return cbar

def add_discrete_cbar_p(fig, ax_target, p_cmap, p_norm, n_p, p_keys, shrink=0.6):
    sm = cm.ScalarMappable(cmap=p_cmap, norm=p_norm)
    sm.set_array([])
    tick_indices = np.arange(0, n_p, 2)
    cbar = fig.colorbar(sm, ax=ax_target, shrink=shrink, pad=0.02,
                        location='right', ticks=tick_indices)
    cbar.set_ticklabels([f'{p_keys[i]:.1f}' for i in tick_indices])
    cbar.set_label(r'$p_{sub}$', fontsize=14)
    return cbar

def get_mean_trajectory_er(model, n, k, field_fn, data, p_ER, q_common):
    cell   = data[model][n][k][p_ER]
    n_iter = len(cell['Betti'])
    trajs  = []
    for i in range(n_iter):
        betti = cell['Betti'][i]
        sc    = cell['SimplexCounts'][i]
        cr    = cell['CycleRank'][i]
        cr_b1 = cell['CycleRank_minus_Betti1'][i]
        sf    = cell['SF'][i]
        fes   = cell['FES'][i]
        if betti is None or sc is None:
            continue
        try:
            traj = field_fn(betti, sc, cr, cr_b1, sf, fes)
            if traj is not None and len(traj) > 0:
                t = np.asarray(traj, dtype=float)
                trajs.append(np.interp(q_common,
                                       np.linspace(0, 1, len(t)), t))
        except Exception:
            continue
    if not trajs:
        return None
    return np.nanmean(trajs, axis=0)

def draw_er_figure(data, base_path, N, k, fields_er, p_ER, q_common, n_colors, n_cmap, n_norm, n_n, model_ls,
                    panels, nrows, ncols, figsize, suptitle, save_prefix,
                    cbar_shrink=0.5, xlim=(0, 1)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()
    all_labels = [label for row in panels for label in row]

    for ax, label in zip(axes_flat, all_labels):
        field_fn = fields_er[label]
        for n in N:
            for model in ['kunif', 'nonuniform']:
                mean = get_mean_trajectory_er(model, n, k, field_fn, data, p_ER, q_common)
                if mean is None:
                    continue
                ax.plot(q_common, mean,
                        color=n_colors[n],
                        ls=model_ls[model],
                        lw=1.5)
        ax.set_ylabel(label)
        ax.set_xlabel(r'Edge density $p$')
        ax.set_xlim(*xlim)

    for ax in axes_flat[len(all_labels):]:
        ax.set_visible(False)

    # Colorbar over N
    add_discrete_cbar_n(fig, axes_flat[:len(all_labels)], n_cmap, n_norm, n_n, N, shrink=cbar_shrink)

    # Manual legend for model linestyles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='grey', ls='-',  lw=1.5, label='K-uniform'),
        Line2D([0], [0], color='grey', ls='--', lw=1.5, label='Nonuniform'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
            frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.045))

    fig.suptitle(suptitle, fontsize=14, y=1.04)
    fig.savefig(base_path + save_prefix + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base_path + save_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_prefix}')

def get_mean_trajectory_ws(data, n_plot, k, field_fn, p_key):
    """
    Returns the mean trajectory (1-D array) over all iterations for a given k,
    interpolated onto a common q grid of length equal to the longest trajectory.
    """
    cell   = data[n_plot][k][p_key]
    n_iter = len(cell['Betti'])
    trajs  = []
    for i in range(n_iter):
        betti = cell['Betti'][i]
        sc    = cell['SimplexCounts'][i]
        euler = cell['Euler'][i]
        cr    = cell['CycleRank'][i]
        cr_b1 = cell['CycleRank_minus_Betti1'][i]
        sf    = cell['SF'][i]
        fes   = cell['FES'][i]
        if betti is None or sc is None:
            continue
        try:
            traj = field_fn(betti, sc, euler, cr, cr_b1, sf, fes)
            if traj is not None and len(traj) > 0:
                trajs.append(np.asarray(traj, dtype=float))
        except Exception:
            continue
    if not trajs:
        return None, None
    max_len  = max(len(t) for t in trajs)
    q_common = np.linspace(0, 1, max_len)
    interp   = np.array([
        np.interp(q_common, np.linspace(0, 1, len(t)), t)
        for t in trajs
    ])
    return q_common, np.nanmean(interp, axis=0)

def draw_ws_figure(data, base_path, n_plot, k, fields_ws, k_colors, k_cmap, k_norm, n_k, K,
                   panels, nrows, ncols, figsize, suptitle, save_prefix,
                   cbar_shrink=0.5):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()
    all_labels = [label for row in panels for label in row]

    for ax, label in zip(axes_flat, all_labels):
        field_fn = fields_ws[label]
        for k in K:
            q, mean = get_mean_trajectory_ws(data, n_plot, k, field_fn, 0.0)
            if q is None:
                continue
            ax.plot(q, mean, color=k_colors[k], lw=1.5)
        ax.set_ylabel(label)
        ax.set_xlabel(r'Rewiring probability $q$')
        ax.set_xlim(0, 1)

    for ax in axes_flat[len(all_labels):]:
        ax.set_visible(False)

    add_discrete_cbar_k(fig, axes_flat[:len(all_labels)], k_cmap, k_norm, n_k, K, shrink=cbar_shrink)

    fig.suptitle(suptitle, fontsize=14, y=1.04)
    fig.savefig(base_path + save_prefix + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base_path + save_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_prefix}')

def draw_simplicial_ws_figure(data, base_path, n_plot, k, fields_ws, p_cmap, p_norm, n_p, p_keys, p_colors,
                              panels, nrows, ncols, figsize, suptitle, save_prefix,
                   cbar_shrink=0.5):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()
    all_labels = [label for row in panels for label in row]

    for ax, label in zip(axes_flat, all_labels):
        field_fn = fields_ws[label]
        for p_key in p_keys:
            q, mean = get_mean_trajectory_ws(data, n_plot, k, field_fn, p_key)
            if q is None:
                continue
            ax.plot(q, mean, color=p_colors[p_key], lw=1.5)
        ax.set_ylabel(label)
        ax.set_xlabel(r'Rewiring Probability $q$')
        ax.set_xlim(0, 1)

    for ax in axes_flat[len(all_labels):]:
        ax.set_visible(False)

    add_discrete_cbar_p(fig, axes_flat[:len(all_labels)], p_cmap, p_norm, n_p, p_keys, shrink=cbar_shrink)

    fig.suptitle(suptitle, fontsize=14, y=1.04)
    fig.savefig(base_path + save_prefix + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base_path + save_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_prefix}')

def mean_term_traj(data, n_plot, n_q, k, q_common, field_fn, p_key):
    """Returns 1-D array of length n_q interpolated onto q_common."""
    cell   = data[n_plot][k][p_key]
    n_iter = len(cell['Betti'])
    trajs  = []
    for i in range(n_iter):
        betti = cell['Betti'][i]
        sc    = cell['SimplexCounts'][i]
        if betti is None or sc is None:
            continue
        try:
            traj = field_fn(betti, sc)
            if traj is not None and len(traj) > 0:
                t = np.asarray(traj, dtype=float)
                trajs.append(np.interp(q_common,
                                       np.linspace(0, 1, len(t)), t))
        except Exception:
            continue
    if not trajs:
        return np.full(n_q, np.nan)
    return np.nanmean(trajs, axis=0)

def get_terminal_mean_nlba(data, n_plot, k, alpha_key, field_fn):
    vals = []
    d    = data[n_plot][k][alpha_key]
    n_it = len(d['Betti'])
    for i in range(n_it):
        betti = d['Betti'][i]
        sc    = d['SimplexCounts'][i]
        euler = d['Euler'][i]
        cr    = d['CycleRank'][i]
        cr_b1 = d['CycleRank_minus_Betti1'][i]
        if betti is None or sc is None:
            continue
        try:
            val = field_fn(betti, sc, euler, cr, cr_b1)
            if val is not None and np.isfinite(val):
                vals.append(float(val))
        except Exception:
            continue
    return np.mean(vals) if vals else np.nan

def draw_nlba_figure(data, base_path, n_plot, k, k_colors, k_cmap, k_norm, n_k, K, alphas, alpha_keys, fields_nlba, panels, nrows, ncols, figsize,
                         suptitle, save_prefix, cbar_shrink=0.5):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    all_labels = [label for row in panels for label in row]

    for ax, label in zip(axes_flat, all_labels):
        field_fn = fields_nlba[label]

        for k in K:
            means = [
                get_terminal_mean_nlba(data, n_plot, k, alpha_key, field_fn)
                for alpha_key in alpha_keys
            ]
            ax.plot(alphas, means,
                    color=k_colors[k],
                    lw=1.5)

        ax.set_ylabel(label)
        ax.set_xlabel(r'$\alpha$ (preferential strength parameter)')
        ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])

    for ax in axes_flat[len(all_labels):]:
        ax.set_visible(False)

    add_discrete_cbar_k(fig, axes_flat[:len(all_labels)], k_cmap, k_norm, n_k, K,
                         shrink=cbar_shrink)

    fig.suptitle(suptitle, fontsize=14, y=1.05)
    fig.savefig(base_path + save_prefix + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base_path + save_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_prefix}')

def add_shared_cbar(fig, axes_list, n_p, p_cmap, p_norm, P_sorted):
    sm = cm.ScalarMappable(cmap=p_cmap, norm=p_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_list, shrink=0.95, pad=0.02,
                        location='right', ticks=np.arange(n_p))
    cbar.set_ticklabels([f'{p:.2f}' for p in P_sorted])
    cbar.set_label(r'$p$ (proportion of new nodes in hyperedge)', fontsize=12)
    return cbar

def analytical_ccdf(d_vals, p):
    alpha  = 1.0 / (1.0 - p)
    log_cc = gammaln(alpha + 1) + gammaln(d_vals + 1) - gammaln(alpha + d_vals + 1)
    return np.exp(log_cc)

def add_slope_line(ax, lam, P_PB_HG_c, P_PB_HG_bins, p_ref=0.45, x_anchor=10.0):
    """
    Draw d^{-gamma} anchored so it passes through the p_ref data curve
    at x = x_anchor.
    """
    key = (lam, p_ref)
    if key not in P_PB_HG_c:
        return
    ccdf_data = np.array(P_PB_HG_c[key], dtype=float)
    bins_data = np.array(P_PB_HG_bins[key], dtype=float)
    mask      = ccdf_data > 0
    d_data    = bins_data[mask]
    cc_data   = ccdf_data[mask]

    # interpolate data value at x_anchor
    if x_anchor < d_data[0] or x_anchor > d_data[-1]:
        return
    y_anchor = np.interp(x_anchor, d_data, cc_data)

    gamma  = 1.0 / (1.0 - p_ref)
    offset = y_anchor * (x_anchor ** gamma)

    x_end  = d_data[-1] * 0.5
    x_vals = np.logspace(np.log10(x_anchor), np.log10(x_end), 100)
    y_vals = offset * x_vals ** (-gamma)

    ax.plot(x_vals, y_vals, color='black', lw=1.2, ls='-.', zorder=5)

    # place annotation at 1/3 along the line, above it
    idx = len(x_vals) // 3
    idx2 = len(x_vals) // 2
    ax.annotate(
        rf'$d^{{-1/(1-p)}} = d^{{-1/(1-{p_ref})}} \approx d^{{-{gamma:.2f}}}$',
        xy = (500,0.005),
        #xy=(x_vals[idx2], y_vals[idx]),
        xytext=(0, 10), textcoords='offset points',
        fontsize=10, va='bottom', ha='center',
    )

def draw_panel(ax, lam, p_subset, p_colors, P_PB_HG_c, P_PB_HG_bins):
    for p in p_subset:
        key = (lam, p)
        if key not in P_PB_HG_c:
            continue
        ccdf_data = np.array(P_PB_HG_c[key], dtype=float)
        bins_data = np.array(P_PB_HG_bins[key], dtype=float)
        mask      = ccdf_data > 0
        d_data    = bins_data[mask]
        cc_data   = ccdf_data[mask]
        ax.plot(d_data, cc_data,
                color=p_colors[p], lw=1.4, ls='-', alpha=0.9)
        ax.plot(d_data, analytical_ccdf(d_data, p),
                color=p_colors[p], lw=1.0, ls='--', alpha=0.9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 1e0)
    ax.set_xlabel(r'Hyperdegree $d$')
    ax.set_ylabel(r'$P(d_H > d)$')

def violin(ax, x, values, color, width=0.35):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return
    vals = np.array(values, dtype=float)
    if len(vals) == 1 or np.unique(vals).size == 1:
        # all values identical — just plot a point
        ax.scatter(x, vals.mean(), color=color, s=60, marker='D',
                   edgecolors='white', linewidths=0.8, zorder=5)
        return
    try:
        kde  = stats.gaussian_kde(vals)
    except np.linalg.LinAlgError:
        ax.scatter(x, vals.mean(), color=color, s=60, marker='D',
                   edgecolors='white', linewidths=0.8, zorder=5)
        return
    yg   = np.linspace(vals.min(), vals.max(), 200)
    dens = kde(yg)
    dens = dens / dens.max() * width
    ax.fill_betweenx(yg, x - dens, x + dens, color=color, alpha=0.25, zorder=2)
    ax.plot(x - dens, yg, color=color, lw=1.0, alpha=0.7, zorder=3)
    ax.plot(x + dens, yg, color=color, lw=1.0, alpha=0.7, zorder=3)
    ax.scatter(x, vals.mean(), color=color, s=50, marker='D',
               edgecolors='white', linewidths=0.8, zorder=5)

# ============================================================================
# CLASSES
# ============================================================================

class TrieNode:
    """Node in a trie for storing hyperedges."""
    __slots__ = ('children', 'timestep')  # Reduce per-instance memory overhead

    def __init__(self):
        # node.children is dict where keys are vertices which are children
        # of the given node and values are TrieNode objects corresponding to the vertices
        self.children = {}
        self.timestep = None  # None if not a complete edge, timestep if complete


class HyperedgeTrie:
    """Trie structure for efficiently checking subset relationships."""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, edge, timestep):
        """Insert an edge (as sorted list of nodes) into the trie."""
        sorted_edge = sorted(edge)  # Sort once
        node = self.root
        for v in sorted_edge:
            if v not in node.children:
                node.children[v] = TrieNode()
            node = node.children[v]
        node.timestep = timestep

    def remove(self, edge):
        """Remove an edge from the trie."""
        sorted_edge = sorted(edge)

        def _remove_recursive(node, depth):
            if depth == len(sorted_edge):
                # We've reached the node representing the edge
                if node.timestep is not None:
                    node.timestep = None
                    # Return True if this node has no children (can be deleted)
                    return len(node.children) == 0
                return False

            v = sorted_edge[depth]
            if v not in node.children:
                return False

            should_delete_child = _remove_recursive(node.children[v], depth + 1)

            if should_delete_child:
                del node.children[v]
                # Return True if this node should also be deleted
                # (no timestep and no children)
                return node.timestep is None and len(node.children) == 0

            return False

        _remove_recursive(self.root, 0)

    def find_subsets(self, sorted_edge):
        """Find all edges in the trie that are subsets of the given edge."""
        subsets = []

        def dfs(node, idx):
            # If this node represents a complete edge, it's a subset
            if node.timestep is not None:
                subsets.append(node.timestep)

            # Try to extend with remaining nodes from edge
            for i in range(idx, len(sorted_edge)):
                v = sorted_edge[i]
                if v in node.children:
                    dfs(node.children[v], i + 1)

        dfs(self.root, 0)
        return subsets


class HypergraphProcessor:
    """
    Process hypergraph using trie for subset checks and node indexing for superset checks.
    Memory-optimized version with __slots__ and compact arrays.
    """
    def __init__(self):
        self.trie = HyperedgeTrie()
        self.edges_by_timestep = {}  # timestep -> frozenset of nodes
        # Use array.array for compact integer storage instead of lists
        self.node_to_edges = defaultdict(lambda: set())

    def add_edge_and_check(self, timestep, edge, H_frozen):
        """
        Add edge and check for subset/superset relationships.

        Args:
            timestep: integer timestep
            edge: set/list of integer node IDs

        Returns:
            dict with timestep, edge, is_superset_of, is_subset_of
        """
        edge_set = frozenset(edge)
        sorted_edge = sorted(edge_set)  # Sort once for reuse

        # Check for subsets using trie
        subsets_of = self.trie.find_subsets(sorted_edge)

        # Check for supersets using node indexing
        supersets_of = self._find_supersets(edge_set)

        # Add edge to trie
        self.trie.insert(sorted_edge, timestep)

        # Track which edges contain each node (using compact arrays)
        for u in edge_set:
            self.node_to_edges[u].add(timestep)

        # Store the edge
        self.edges_by_timestep[timestep] = edge_set

        return timestep, sorted_edge, subsets_of, supersets_of

    def remove_edge_and_check(self, timestep, edge, H_frozen):
        """
        Remove edge and get its subset/superset relationships for cleanup.

        Args:
            timestep: integer timestep
            edge: set/list of integer node IDs

        Returns:
            tuple with timestep, sorted_edge, subsets_of, supersets_of
        """
        edge_set = frozenset(edge)
        sorted_edge = sorted(edge_set)

        # Get subsets and supersets before removal
        subsets_of = self.trie.find_subsets(sorted_edge)
        for idx in subsets_of:
            if H_frozen[idx] == edge_set:
                timestep_original = idx
                subsets_of.remove(idx)
                break
        supersets_of = self._find_supersets(edge_set)
        upper_pairs_removed = 0; lower_pairs_removed = 0
        # If the subset idx < timestep_original then the smaller edge was
        # added first, forming an upper pair, so removing the edge destroys
        # the upper pair.
        for idx in subsets_of:
            if idx < timestep_original:
                upper_pairs_removed += 1
            else:
                lower_pairs_removed += 1

        # If the superset idx < timestep_original then the larger edge was
        # added first, forming a lower pair, so removing the edge destroys
        # the lower pair.
        for idx in supersets_of:
            if idx < timestep_original:
                lower_pairs_removed += 1
            else:
                upper_pairs_removed += 1

        # Remove edge from trie
        self.trie.remove(sorted_edge)

        # Remove from node indexing
        for u in edge_set:
            if u in self.node_to_edges:
                if timestep_original in self.node_to_edges[u]:
                    self.node_to_edges[u].remove(timestep_original)

        return timestep, sorted_edge, subsets_of, supersets_of, upper_pairs_removed, lower_pairs_removed

    def _find_supersets(self, edge):
        """
        Find all previous edges that are supersets of the given edge.
        Uses node indexing to find candidate edges efficiently.
        """
        if len(edge) == 0:
            return []

        # Find candidate edges: those that contain ALL nodes from edge
        edge_list = list(edge)

        # Start with edges containing the first node
        # Convert array to set for efficient intersection
        candidate_timesteps = set(self.node_to_edges[edge_list[0]])

        # Intersect with edges containing each remaining node
        for v in edge_list[1:]:
            candidate_timesteps &= set(self.node_to_edges[v])

        # Filter candidates to only those that are proper supersets
        supersets = []
        for t in candidate_timesteps:
            candidate_edge = self.edges_by_timestep[t]
            if len(candidate_edge) > len(edge):
                supersets.append(t)

        return supersets

class WeightedSampler:
    """Efficient data structure for weighted sampling with dynamic updates"""

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