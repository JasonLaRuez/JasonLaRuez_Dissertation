"""
Module Name: GenerativeHypergraphModels.py
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

import matplotlib.pyplot as plt # Plotting
import time # Timing simulations
from tqdm.notebook import trange, tqdm # Allows for real-time progress bar of simulations

import gc # Memory management
import pickle # Takes environment variables and saves them as is
import gzip # Allows for compression of saved files
from joblib import Parallel, delayed # Parallelization functions
import multiprocessing # Get number of cpu cores
import collections # Collecting degree dist using counter

import math
from collections import defaultdict
from array import array
import bisect

# Local imports (other modules you've created)
# from my_other_module import some_function


# ============================================================================
# SETTING CONSTANTS
# ============================================================================


# ============================================================================
# HELPER FUNCTIONS (Private)
# ============================================================================
def _private_helper_function(x):
    """
    Private helper function (prefix with underscore).
    These won't be imported with 'from module import *'
    
    Args:
        x: Input parameter
        
    Returns:
        Processed result
    """
    return x * 2


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

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
    Input: m - integer corresponding the max number of nodes to add at
               each timestep. At each timestep a number 1 <= size <= m
               is picked uniformly at random, and that many nodes are
               added to the network contained within a hyperedge of size
               "size" + 1.
           steps - integer number of iterations for which to run the model.
               Hypergraph ends with "steps" many hyperedges and [(m+1)/2]*steps + 1
               nodes in the expectation, and total degree steps*([(m+1)*(m+2)/2-1]/m)
               in the expectation.
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

def HG_NL_PreferentialAttachment_kUnif(K, steps, alpha = 1.0, timing=False):
    """
    Input: m - integer corresponding the max number of nodes to add at
               each timestep. At each timestep a number 1 <= size <= m
               is picked uniformly at random, and that many nodes are
               added to the network contained within a hyperedge of size
               "size" + 1.
           steps - integer number of iterations for which to run the model.
               Hypergraph ends with "steps" many hyperedges and [(m+1)/2]*steps + 1
               nodes in the expectation, and total degree steps*([(m+1)*(m+2)/2-1]/m)
               in the expectation.
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

# ============================================================================
# CLASSES
# ============================================================================
class DataProcessor:
    """
    A class for processing and analyzing data.
    
    Attributes:
        name (str): Name of the processor
        config (dict): Configuration parameters
    """
    
    def __init__(self, name, config=None):
        """
        Initialize the DataProcessor.
        
        Args:
            name (str): Processor name
            config (dict, optional): Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._results = []  # Private attribute
    
    def process(self, data):
        """
        Process input data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        # Your processing logic here
        processed = data
        self._results.append(processed)
        return processed
    
    def get_results(self):
        """Return all stored results."""
        return self._results.copy()
    
    def __repr__(self):
        """String representation of the object."""
        return f"DataProcessor(name='{self.name}')"