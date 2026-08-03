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

import matplotlib.pyplot as plt # Plotting
import time # Timing simulations
from tqdm.notebook import trange, tqdm # Allows for real-time progress bar of simulations

import gc # Memory management
import pickle # Takes environment variables and saves them as is
import gzip # Allows for compression of saved files
from joblib import Parallel, delayed # Parallelization functions
import multiprocessing # Get number of cpu cores

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
# Rewire-to-random on Networks
# ============================================================================
def RewireToRandomVoter(G, rho, alpha, timing = False):
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

    # Set density of opinion 0 to rho, and 1 to (1 - rho)
    N = len(G.nodes())
    Opinions = np.array([0 if i < int(N*rho) else 1 for i in range(N)])

    # Generate list of all edges where the connected nodes have differing (discordant) opinions
    DiscordantEdges = [edge for edge in G.edges() if Opinions[edge[0]] != Opinions[edge[1]]]

    # Initialize list of opinion proportions
    Proportions = [sum(Opinions)/N]
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

    while len(DiscordantEdges) > 0:
        # Proportions from previous time step, and update timer
        timer += 1; Proportions.append(Proportions[-1])

        # Uniformly select a discordant edge
        edgeChoice = np.random.choice(len(DiscordantEdges))
        edge = DiscordantEdges[edgeChoice]

        # Choose either 0 or 1 to choose which node in the edge is the source and which is the target
        choice = np.random.choice(2)
        source = edge[choice]
        target = edge[(choice + 1) % 2]

        # Rewiring (probability alpha)
        if random.random() < alpha:
            # Remove edge from G
            G.remove_edge(source, target)

            # Since we want to remove edge, which is at index edgeChoice in list,
            # we move the last element of the list into index edgeChoice, and pop
            # the last element, reducing the remove edge complexity from O(n) to O(1).
            DiscordantEdges[edgeChoice] = DiscordantEdges[-1]
            DiscordantEdges.pop()

            while True:
                # Randomly select new target
                newTarget = np.random.choice(N)

                # Check that newTarget is not already connected to source, else draw a different newTarget
                # Since the average degree of a node is low, this should be faster than a deterministic selection
                if (newTarget not in G.neighbors(source)) and (newTarget != source) and (newTarget != target):
                    break

            # Add edge (source,newTarget) to G
            G.add_edge(source,newTarget);

            # Now that the edge has been rewired, check if it is now discordant
            if Opinions[source] != Opinions[newTarget]:
                DiscordantEdges.append( tuple(sorted([source, newTarget])) )

        # Opinion adoption (probability 1 - alpha)
        else:
            # Source node adopts opinion of target node
            Opinions[source] = Opinions[target]

            # Either add 1/n or subtract 1/n to proportion of 1's
            if Opinions[target] == 1:
                Proportions[timer] += 1/N
            else:
                Proportions[timer] -= 1/N

            # Since we have changed the opinion of source node, we need to update whether edges containing
            # source node are discordant or not
            for neighbor in G.neighbors(source):
                # If the opinions of the source and its neighbor differ, then they previously
                # were the same and thus not discordant, must add edge to discordant list
                if Opinions[source] != Opinions[neighbor]:
                    DiscordantEdges.append( tuple(sorted([source,neighbor])) )

                # If the opinions of the source and its neighbor are the same, then they
                # previously were discordant and so we must remove the edge from the discordant list
                else:
                    DiscordantEdges.remove( tuple(sorted([source,neighbor])) )

    if timing == True:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)

    return Proportions, G

def ZZPH_RewireToRandomVoter(G, rho, alpha, timing = False):
    """
    Input a networkx object G, initial opinion 0 density rho, and
    rewiring probability alpha. Simulate the adaptive network voter
    model on the input graph, where at each step a discordant edge
    is selected uniformly. Then, with probability alpha the edge is rewired
    at random, and with probability 1-alpha one node adopts the opinion
    of its neighbor. Returns the sets of the 0-th, 1-st, 2-nd and 3-rd
    betti numbers, using persistent homology (3-rd betti number is
    truncated betti number), the counts of different dimensional simplices
    at each time step, the set of Euler characteristics, and proportions
    of opinions at each time step.
    """

    if timing == True:
        print("(1/5) Initializing graph, variables and data structures",flush=True)
        start = time.time()

    # Set density of opinion 0 to rho, and 1 to (1 - rho)
    N = len(G.nodes())
    Opinions = np.array([0 if i < int(N*rho) else 1 for i in range(N)])

    # Generate list of all edges where the connected nodes have differing (discordant) opinions
    DiscordantEdges = [edge for edge in G.edges() if Opinions[edge[0]] != Opinions[edge[1]]]

    # Initialize list of opinion proportions
    Proportions = [sum(Opinions)/N]

    # Initialize dictionary which keeps track of when simplices
    # are added and removed, and list of simplex counts
    Cliques = nx.enumerate_all_cliques(G)
    Times = {tuple(clique) : [0] for clique in Cliques}
    SimplexCounts = [ [0] * max(len(c) for c in nx.find_cliques(G)) ]
    for simplex in Times:
        SimplexCounts[0][len(simplex)-1] += 1

    timer = 0

    if timing == True:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    # Main loop of the model. At each step select a discordant edge at random
    # With probability alpha, the (randomly selected) source node is rewired from the target node to a random
    # node with the same opinion as the source node, and is not already connected to the source node
    # Otherwise (with probability 1 - alpha) the source node adopts the opinion of the target node

    while len(DiscordantEdges) > 0:
        # Copy simplex counts and proportions from previous time step, and update timer
        timer += 1; SimplexCounts.append(SimplexCounts[-1].copy());
        Proportions.append(Proportions[-1])

        # Uniformly select a discordant edge
        edgeChoice = np.random.choice(len(DiscordantEdges))
        edge = DiscordantEdges[edgeChoice]

        # Choose either 0 or 1 to choose which node in the edge is the source and which is the target
        choice = np.random.choice(2)
        source = edge[choice]
        target = edge[(choice + 1) % 2]

        # Rewiring (probability alpha)
        if random.random() < alpha:
          # Removing the edge (source, target) removes simplices, so we find these
          # simplices and remove them. We use a set() for simplices to avoid double
          # adding simplices from maxial cliques, and initialize Simplices to
          # contain (source,target) to reduce need for computation
          Simplices = set([tuple(sorted([source,target]))]); Cliques = nx.find_cliques(G,sorted([source, target])) # <- Returns maximal cliques containing e
          for clique in Cliques:
              numNodes = len(clique)
              # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
              # all the way down to the 3-subsets (triangles). We don't do edges or nodes
              # since we never add/remove nodes, and the only edge we care about has
              # already been added. This reduces computation.
              # We do sorted() to avoid double creating simplices
              clique = sorted(clique)
              for r in range(numNodes, 2, -1):
                  # combinations(S,r) is from itertools and returns iterator corresponding
                  # to all r-subsets of S.
                  for face in combinations(clique, r):
                      # Only add simplices containing the edge e
                      if (source in face) and (target in face):
                          Simplices.add(face)

          # From set of simplices extract simplex counts, and if the simplex
          # is a tetrahedron or smaller add it to Times
          for simplex in Simplices:
              SimplexCounts[-1][len(simplex)-1] -= 1
              if len(simplex) <= 4:
                  # Since we are removing the simplex, it must already exist in Times dict
                  Times[simplex].append(timer)

          # Remove edge from G
          G.remove_edge(source, target)

          # Since we want to remove edge, which is at index edgeChoice in list,
          # we move the last element of the list into index edgeChoice, and pop
          # the last element, reducing the remove edge complexity from O(n) to O(1).
          DiscordantEdges[edgeChoice] = DiscordantEdges[-1]
          DiscordantEdges.pop()

          while True:
              # Randomly select new target
              newTarget = np.random.choice(N)

              # Check that newTarget is not already connected to source, else draw a different newTarget
              # Since the average degree of a node is low, this should be faster than a deterministic selection
              if (newTarget not in G.neighbors(source)) and (newTarget != source) and (newTarget != target):
                  break

          # Add edge (source,newTarget) to G
          G.add_edge(source,newTarget);

          # Find all of the newly added simplices (which must contain source and newTarget by necessity)
          # We use a set() for simplices to avoid double adding simplices from maxial cliques,
          # and initialize Simplices to contain (source,newTarget) to reduce need for computation
          Simplices = set([tuple(sorted([source,newTarget]))]); Cliques = nx.find_cliques(G,sorted([source,newTarget])) # <- Returns maximal cliques containing source and newTarget
          for clique in Cliques:
              numNodes = len(clique)
              # If a newly added simplex is larger than the previously largest simplex,
              # it can only be one larger so be increase the size of simplex counts by one
              if numNodes > len(SimplexCounts[-1]):
                  SimplexCounts[-1].append(0)
              # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
              # all the way down to the 3-subsets (triangles). We don't do edges or nodes
              # since we never add/remove nodes, and the only edge we care about has
              # already been added. This reduces computation.
              clique = sorted(clique)
              for r in range(numNodes, 2, -1):
                  # combinations(S,r) is from itertools and returns iterator corresponding
                  # to all r-subsets of S.
                  for face in combinations(clique, r):
                      if (source in face) and (newTarget in face):
                          Simplices.add(face)

          # From set of simplices extract simplex counts, and if the simplex
          # is a tetrahedron or smaller add it to Times
          for simplex in Simplices:
              SimplexCounts[-1][len(simplex)-1] += 1
              if len(simplex) <= 4:
                  # Since we are adding simplices to complex, we don't know if they
                  # previously exists and were then removed, i.e. we need to check if
                  # they already exist in Times dict
                  if simplex in Times:
                      Times[simplex].append(timer)
                  else:
                      Times[simplex] = [timer]

          # Now that the edge has been rewired, check if it is now discordant
          if Opinions[source] != Opinions[newTarget]:
              DiscordantEdges.append( tuple(sorted([source, newTarget])) )

        # Opinion adoption (probability 1 - alpha)
        else:
            # Source node adopts opinion of target node
            Opinions[source] = Opinions[target]

            # Either add 1/n or subtract 1/n to proportion of 1's
            if Opinions[target] == 1:
                Proportions[timer] += 1/N
            else:
                Proportions[timer] -= 1/N

            # Since we have changed the opinion of source node, we need to update whether edges containing
            # source node are discordant or not
            for neighbor in G.neighbors(source):
                # If the opinions of the source and its neighbor differ, then they previously
                # were the same and thus not discordant, must add edge to discordant list
                if Opinions[source] != Opinions[neighbor]:
                    DiscordantEdges.append( tuple(sorted([source,neighbor])) )

                # If the opinions of the source and its neighbor are the same, then they
                # previously were discordant and so we must remove the edge from the discordant list
                else:
                    DiscordantEdges.remove( tuple(sorted([source,neighbor])) )

    if timing == True:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Zigzag Persistent Homology",flush=True)
        start = time.time()

    # Extract list of every simplex added/removed, and list of times they were
    # added/removed, for input into zigzag persistence.
    simplices = [list(key) for key in Times]; times = [Times[key] for key in Times]

    # Clear out Times, which is massive
    del(Times); del(G); gc.collect()

    # Construct filtration and compute homology
    f = d.Filtration(simplices)
    zz, dgms, cells = d.zigzag_homology_persistence(f, times)

    # Clear out remaining lists which are massive
    del(simplices); del(times); gc.collect()

    if timing == True:
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

    if timing == True:
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

    if timing == True:
        end = time.time()
        print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds")

    return Betti, SimplexCounts, Euler, Proportions

def ParallelCall_RewireRandom(params):
    G = params[0]; rho = params[1]; alpha = params[2];
    Betti, SimplexCounts, Euler, Proportions = ZZPH_RewireToRandomVoter(G, rho, alpha)
    Data = [Betti, SimplexCounts, Euler, Proportions]
    return Data

def Parallel_RewireRandom(G, rhos, alphas):
    # Set N to be  number of nodes, and m the number of edges
    # Set rhos to be the set of densities of opinions
    # Set alphas to be the rewiring probability (social selection vs influence)
    params = [[G], rhos, alphas]
    params = [p for p in product(*params)]

    # Get number of cpus that can be used
    num_cores = multiprocessing.cpu_count()

    # Actual call to parallel function
    results = Parallel(n_jobs=num_cores)(delayed(ParallelCall_RewireRandom)(param) for param in tqdm(params))

# ============================================================================
# Rewire-to-same on Networks
# ============================================================================

def ZZPH_RewireToSameVoter(G, rho, alpha, timing = False):
    """
    Input a networkx object G, initial opinion 0 density rho, and
    rewiring probability alpha. Simulate the adaptive network voter
    model on the input graph, where at each step a discordant edge
    is selected uniformly. Then, with probability alpha the edge is rewired
    from a source node to a new target node with the same opinion as the
    source node, and with probability 1-alpha the source node adopts the opinion
    of its neighbor. Returns the sets of the 0-th, 1-st, 2-nd and 3-rd
    betti numbers, using persistent homology (3-rd betti number is
    truncated betti number), the counts of different dimensional simplices
    at each time step, the set of Euler characteristics, and proportions
    of opinions at each time step.
    """

    if timing == True:
        print("(1/5) Initializing graph, variables and data structures",flush=True)
        start = time.time()

    N = len(G.nodes())

    # Set density of opinion 0 to rho, and 1 to 1 - rho
    # Opinions[0] and Opinions[1] contain lists of nodes with opinions 0 and 1 respectively, Opinions[2][i] contains opinion of node i
    Opinions = [ list(range(int(N*rho))), list(range(int(N*rho),N)), np.array([0 if i < int(N*rho) else 1 for i in range(N)]) ]

    # Generate list of all edges where the connected nodes have differing (discordant) opinions
    DiscordantEdges = [edge for edge in G.edges() if Opinions[2][edge[0]] != Opinions[2][edge[1]]]

    # Initialize list for proportions
    Proportions = [sum(Opinions[2])/N]

    # Initialize dictionary which keeps track of when simplices
    # are added and removed, and list of simplex counts
    Cliques = nx.enumerate_all_cliques(G)
    Times = {tuple(clique) : [0] for clique in Cliques}
    SimplexCounts = [ [0] * max(len(c) for c in nx.find_cliques(G)) ]
    for simplex in Times:
        SimplexCounts[0][len(simplex)-1] += 1

    timer = 0

    if timing == True:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    # Main loop of the model. At each step select a discordant edge at random
    # With probability alpha, the (randomly selected) source node is rewired from the target node to a random
    # node with the same opinion as the source node, and is not already connected to the source node
    # Otherwise (with probability 1 - alpha) the source node adopts the opinion of the target node

    while len(DiscordantEdges) > 0:
        # Copy simplex counts and proportions from previous time step, and update timer
        timer += 1; SimplexCounts.append(SimplexCounts[-1].copy());
        Proportions.append(Proportions[-1])

        # Uniformly select a discordant edge
        edgeChoice = np.random.choice(len(DiscordantEdges))
        edge = DiscordantEdges[edgeChoice]

        # Choose either 0 or 1 to choose which node in the edge is the source and which is the target
        choice = np.random.choice(2)
        source = edge[choice]
        target = edge[(choice + 1) % 2]

        # Rewiring (probability alpha)
        if random.random() < alpha:
          # Check if rewiring is possible (rewiring is not possible if all nodes with the same opinion as the source are already connected to the source)
          if len(Opinions[Opinions[2][source]]) <= len([neighbor for neighbor in G.neighbors(source) if Opinions[2][source] == Opinions[2][neighbor]]) + 1:
              continue

          # Removing the edge (source, target) removes simplices, so we find these
          # simplices and remove them. We use a set() for simplices to avoid double
          # adding simplices from maxial cliques, and initialize Simplices to
          # contain (source,target) to reduce need for computation
          Simplices = set([tuple(sorted([source,target]))]); Cliques = nx.find_cliques(G,sorted([source, target])) # <- Returns maximal cliques containing e
          for clique in Cliques:
              numNodes = len(clique)
              # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
              # all the way down to the 3-subsets (triangles). We don't do edges or nodes
              # since we never add/remove nodes, and the only edge we care about has
              # already been added. This reduces computation.
              # We do sorted() to avoid double creating simplices
              clique = sorted(clique)
              for r in range(numNodes, 2, -1):
                  # combinations(S,r) is from itertools and returns iterator corresponding
                  # to all r-subsets of S.
                  for face in combinations(clique, r):
                      # Only add simplices containing the edge e
                      if (source in face) and (target in face):
                          Simplices.add(face)

          # From set of simplices extract simplex counts, and if the simplex
          # is a tetrahedron or smaller add it to Times
          for simplex in Simplices:
              SimplexCounts[-1][len(simplex)-1] -= 1
              if len(simplex) <= 4:
                  # Since we are removing the simplex, it must already exist in Times dict
                  Times[simplex].append(timer)

          # Remove edge from G
          G.remove_edge(source, target)

          # Since we want to remove edge, which is at index edgeChoice in list,
          # we move the last element of the list into index edgeChoice, and pop
          # the last element, reducing the remove edge complexity from O(n) to O(1).
          DiscordantEdges[edgeChoice] = DiscordantEdges[-1]
          DiscordantEdges.pop()

          # Select new target node to wire source to from set of nodes with same opinion as source
          while True:
              newTarget = np.random.choice(Opinions[Opinions[2][source]])

              # Check that newTarget is not already connected to source, else draw a different newTarget
              # Since the average degree of a node is low, this should be faster than a deterministic selection
              if (newTarget not in G.neighbors(source)) and (newTarget != source) and (newTarget != target):
                  break

          # Add edge (source,newTarget) to G
          G.add_edge(source,newTarget);

          # Find all of the newly added simplices (which must contain source and newTarget by necessity)
          # We use a set() for simplices to avoid double adding simplices from maxial cliques,
          # and initialize Simplices to contain (source,newTarget) to reduce need for computation
          Simplices = set([tuple(sorted([source,newTarget]))]); Cliques = nx.find_cliques(G,sorted([source,newTarget])) # <- Returns maximal cliques containing source and newTarget
          for clique in Cliques:
              numNodes = len(clique)
              # If a newly added simplex is larger than the previously largest simplex,
              # it can only be one larger so be increase the size of simplex counts by one
              if numNodes > len(SimplexCounts[-1]):
                  SimplexCounts[-1].append(0)
              # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
              # all the way down to the 3-subsets (triangles). We don't do edges or nodes
              # since we never add/remove nodes, and the only edge we care about has
              # already been added. This reduces computation.
              clique = sorted(clique)
              for r in range(numNodes, 2, -1):
                  # combinations(S,r) is from itertools and returns iterator corresponding
                  # to all r-subsets of S.
                  for face in combinations(clique, r):
                      if (source in face) and (newTarget in face):
                          Simplices.add(face)

          # From set of simplices extract simplex counts, and if the simplex
          # is a tetrahedron or smaller add it to Times
          for simplex in Simplices:
              SimplexCounts[-1][len(simplex)-1] += 1
              if len(simplex) <= 4:
                  # Since we are adding simplices to complex, we don't know if they
                  # previously exists and were then removed, i.e. we need to check if
                  # they already exist in Times dict
                  if simplex in Times:
                      Times[simplex].append(timer)
                  else:
                      Times[simplex] = [timer]

        # Opinion adoption (probability 1 - alpha)
        else:
            # Source node adopts opinion of target node
            Opinions[Opinions[2][source]].remove(source)
            Opinions[2][source] = Opinions[2][target]
            Opinions[Opinions[2][source]].append(source)

            # Either add 1/n or subtract 1/n to proportion of 1's
            if Opinions[2][target] == 1:
                Proportions[timer] += 1/N
            else:
                Proportions[timer] -= 1/N

            # Since we have changed the opinion of source node, we need to update whether edges containing
            # source node are discordant or not
            for neighbor in G.neighbors(source):
                # If the opinions of the source and its neighbor differ, then they previously
                # were the same and thus not discordant, must add edge to discordant list
                if Opinions[2][source] != Opinions[2][neighbor]:
                    DiscordantEdges.append( tuple(sorted([source, neighbor])) )

                # If the opinions of the source and its neighbor are the same, then they
                # previously were discordant and so we must remove the edge from the discordant list
                else:
                    DiscordantEdges.remove( tuple(sorted([source, neighbor])) )

    if timing == True:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Zigzag Persistent Homology",flush=True)
        start = time.time()

    # Extract list of every simplex added/removed, and list of times they were
    # added/removed, for input into zigzag persistence.
    simplices = [list(key) for key in Times]; times = [Times[key] for key in Times]

    # Clear out Times, which is massive
    del(Times); del(G); gc.collect()

    # Construct filtration and compute homology
    f = d.Filtration(simplices)
    zz, dgms, cells = d.zigzag_homology_persistence(f, times)

    # Clear out remaining lists which are massive
    del(simplices); del(times); gc.collect()

    if timing == True:
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

    if timing == True:
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

    if timing == True:
        end = time.time()
        print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds")

    return Betti, SimplexCounts, Euler, Proportions

def ParallelCall_RewireSame(params):
    """
    """
    G = params[0]; rho = params[1]; alpha = params[2];
    Betti, SimplexCounts, Euler, Proportions = ZZPH_RewireToSameVoter(G, rho, alpha)
    Data = [Betti, SimplexCounts, Euler, Proportions]
    return 0

def Parallel_RewireSame(G, rhos, alphas):
    """
    """
    # Set rhos to be the set of initial densities of opinions
    # Set alphas to be the rewiring probability (social selection vs influence)
    params = [[G], rhos, alphas]
    params = [p for p in product(*params)]

    # Get number of cpus that can be used
    num_cores = multiprocessing.cpu_count()

    # Actual call to parallel function
    results = Parallel(n_jobs=num_cores)(delayed(ParallelCall_RewireSame)(param) for param in tqdm(params))

# ============================================================================
# Transitivity rewire on Networks
# ============================================================================

def ZZPH_TriangleRewireVoter(G, rho, alpha, gamma, timing = False):
    """
    Input a networkx object G, initial opinion 0 density rho, rewiring probability
    alpha, and triangle closing probability gamma. Simulate the adaptive network
    voter model on the input graph, where at each step a discordant edge
    is selected uniformly. Then, with probability alpha the edge is rewired
    from a source node to a new target node. With probability gamma the new
    target node is a neighbor of a neighbor of the source node, closing the
    triangle, otherwise (with probability 1-gamma) the new target is selected
    randomly. With probability 1-alpha the source node adopts the opinion
    of its neighbor. Returns the sets of the 0-th, 1-st, 2-nd and 3-rd
    betti numbers, using persistent homology (3-rd betti number is
    truncated betti number), the counts of different dimensional simplices
    at each time step, the set of Euler characteristics, and proportions
    of opinions at each time step.
    """

    if timing == True:
        print("(1/5) Initializing graph, variables and data structures",flush=True)
        start = time.time()

    N = len(G.nodes())

    # Set density of opinion 0 to rho, and 1 to 1 - rho
    Opinions = np.array([0 if i < int(N*rho) else 1 for i in range(N)])

    # Generate list of all edges where the connected nodes have differing (discordant) opinions
    DiscordantEdges = [edge for edge in G.edges() if Opinions[edge[0]] != Opinions[edge[1]]]

    # Initialize list for proportions
    Proportions = [sum(Opinions)/N]

    # Initialize dictionary which keeps track of when simplices
    # are added and removed, and list of simplex counts
    Cliques = nx.enumerate_all_cliques(G)
    Times = {tuple(clique) : [0] for clique in Cliques}
    SimplexCounts = [ [0] * max(len(c) for c in nx.find_cliques(G)) ]
    for simplex in Times:
      SimplexCounts[0][len(simplex)-1] += 1

    timer = 0

    if timing == True:
        end = time.time()
        print("Initialization complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(2/5) Beginning network evolution",flush=True)
        start = time.time()

    # Main loop of the model. At each step select a discordant edge at random
    # With probability alpha, the (randomly selected) source node is rewired from the target node to
    # neighbor of one of its neighbors with probability gamma, or to a random
    # node with probability 1-gamma. Otherwise (with probability 1 - alpha)
    # the source node adopts the opinion of the target node
    while len(DiscordantEdges) > 0:
        # Copy simplex counts and proportions from previous time step, and update timer
        timer += 1; SimplexCounts.append(SimplexCounts[-1].copy());
        Proportions.append(Proportions[-1])

        # Uniformly select a discordant edge
        edgeChoice = np.random.choice(len(DiscordantEdges))
        edge = DiscordantEdges[edgeChoice]

        # Choose either 0 or 1 to choose which node in the edge is the source and which is the target
        choice = np.random.choice(2)
        source = edge[choice]
        target = edge[(choice + 1) % 2]

        # Rewiring (probability alpha)
        if random.random() < alpha:
            # Removing the edge (source, target) removes simplices, so we find these
            # simplices and remove them. We use a set() for simplices to avoid double
            # adding simplices from maxial cliques, and initialize Simplices to
            # contain (source,target) to reduce need for computation
            Simplices = set([tuple(sorted([source,target]))]); Cliques = nx.find_cliques(G,sorted([source, target])) # <- Returns maximal cliques containing e
            for clique in Cliques:
                numNodes = len(clique)
                # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
                # all the way down to the 3-subsets (triangles). We don't do edges or nodes
                # since we never add/remove nodes, and the only edge we care about has
                # already been added. This reduces computation.
                # We do sorted() to avoid double creating simplices
                clique = sorted(clique)
                for r in range(numNodes, 2, -1):
                    # combinations(S,r) is from itertools and returns iterator corresponding
                    # to all r-subsets of S.
                    for face in combinations(clique, r):
                        # Only add simplices containing the edge e
                        if (source in face) and (target in face):
                            Simplices.add(face)

            # From set of simplices extract simplex counts, and if the simplex
            # is a tetrahedron or smaller add it to Times
            for simplex in Simplices:
                SimplexCounts[-1][len(simplex)-1] -= 1
                if len(simplex) <= 4:
                    # Since we are removing the simplex, it must already exist in Times dict
                    Times[simplex].append(timer)

            # Remove edge from G
            G.remove_edge(source, target)

            # Since we want to remove edge, which is at index edgeChoice in list,
            # we move the last element of the list into index edgeChoice, and pop
            # the last element, reducing the remove edge complexity from O(n) to O(1).
            DiscordantEdges[edgeChoice] = DiscordantEdges[-1]
            DiscordantEdges.pop()

            # With probability gamma randomly select the new target node to be a neighbor
            # of a neighbor of the source node. If no valid selection exists, default
            # to random selection.
            if random.random() < gamma:
                # Construct the list of neighbors of neighbors of the source node, not including
                # the source nodes neighbors, the source node itself, and any duplicate nodes.
                NeighborsOfNeighbors = list(set([NofN for neighbor in G.neighbors(source) for NofN in G.neighbors(neighbor) if (NofN not in G.neighbors(source) and NofN != source and NofN != target)]))

                # Check that a valid neighbor of neighbor exists
                if len(NeighborsOfNeighbors) > 0:
                    newTarget = np.random.choice(NeighborsOfNeighbors)
                # Otherwise resort to random selection
                else:
                    while True:
                        newTarget = np.random.choice(N)
                        # Check that newTarget is not already connected to source, else draw a different newTarget
                        # Since the average degree of a node is low, this should be faster than a deterministic selection
                        if (newTarget not in G.neighbors(source)) and (newTarget != source) and (newTarget != target):
                            break

            # With probability 1-gamma, random selection
            else:
                while True:
                    newTarget = np.random.choice(N)
                    # Check that newTarget is not already connected to source, else draw a different newTarget
                    # Since the average degree of a node is low, this should be faster than a deterministic selection
                    if (newTarget not in G.neighbors(source)) and (newTarget != source) and (newTarget != target):
                        break

            # Add edge (source,newTarget) to G
            G.add_edge(source,newTarget);

            # Now that the edge has been rewired, check if it is now discordant
            if Opinions[source] != Opinions[newTarget]:
                DiscordantEdges.append( tuple(sorted([source, newTarget])) )

            # Find all of the newly added simplices (which must contain source and newTarget by necessity)
            # We use a set() for simplices to avoid double adding simplices from maxial cliques,
            # and initialize Simplices to contain (source,newTarget) to reduce need for computation
            Simplices = set([tuple(sorted([source,newTarget]))]); Cliques = nx.find_cliques(G,sorted([source,newTarget])) # <- Returns maximal cliques containing source and newTarget
            for clique in Cliques:
                numNodes = len(clique)
                # If a newly added simplex is larger than the previously largest simplex,
                # it can only be one larger so be increase the size of simplex counts by one
                if numNodes > len(SimplexCounts[-1]):
                    SimplexCounts[-1].append(0)
                # For a clique of n nodes, include the clique itself, its n-1 subsets, n-2 subsets,
                # all the way down to the 3-subsets (triangles). We don't do edges or nodes
                # since we never add/remove nodes, and the only edge we care about has
                # already been added. This reduces computation.
                clique = sorted(clique)
                for r in range(numNodes, 2, -1):
                    # combinations(S,r) is from itertools and returns iterator corresponding
                    # to all r-subsets of S.
                    for face in combinations(clique, r):
                        if (source in face) and (newTarget in face):
                            Simplices.add(face)

            # From set of simplices extract simplex counts, and if the simplex
            # is a tetrahedron or smaller add it to Times
            for simplex in Simplices:
                SimplexCounts[-1][len(simplex)-1] += 1
                if len(simplex) <= 4:
                    # Since we are adding simplices to complex, we don't know if they
                    # previously exists and were then removed, i.e. we need to check if
                    # they already exist in Times dict
                    if simplex in Times:
                        Times[simplex].append(timer)
                    else:
                        Times[simplex] = [timer]

        # Opinion adoption (probability 1 - alpha)
        else:
            # Source node adopts opinion of target node
            Opinions[source] = Opinions[target]

            # Either add 1/n or subtract 1/n to proportion of 1's
            if Opinions[target] == 1:
                Proportions[timer] += 1/N
            else:
                Proportions[timer] -= 1/N

            # Since we have changed the opinion of source node, we need to update whether edges containing
            # source node are discordant or not
            for neighbor in G.neighbors(source):
                # If the opinions of the source and its neighbor differ, then they previously
                # were the same and thus not discordant, must add edge to discordant list
                if Opinions[source] != Opinions[neighbor]:
                    DiscordantEdges.append( tuple(sorted([source, neighbor])) )

                # If the opinions of the source and its neighbor are the same, then they
                # previously were discordant and so we must remove the edge from the discordant list
                else:
                    DiscordantEdges.remove( tuple(sorted([source, neighbor])) )

    if timing == True:
        end = time.time()
        print("Evolution complete, time taken : "+str(end - start)+" seconds",flush=True)
        print("(3/5) Beginning Zigzag Persistent Homology",flush=True)
        start = time.time()

    # Extract list of every simplex added/removed, and list of times they were
    # added/removed, for input into zigzag persistence.
    simplices = [list(key) for key in Times]; times = [Times[key] for key in Times]

    # Clear out Times, which is massive
    del(Times); del(G); gc.collect()

    # Construct filtration and compute homology
    f = d.Filtration(simplices)
    zz, dgms, cells = d.zigzag_homology_persistence(f, times)

    # Clear out remaining lists which are massive
    del(simplices); del(times); gc.collect()

    if timing == True:
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

    if timing == True:
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

    if timing == True:
        end = time.time()
        print("Euler Characteristic extraction complete, time taken : "+str(end - start)+" seconds")

    return Betti, SimplexCounts, Euler, Proportions


def ParallelCall_RewireTriangle(params):
    """
    """
    G = params[0]; rho = params[1]; alpha = params[2]; gamma = params[3]
    Betti, SimplexCounts, Euler, Proportions = ZZPH_TriangleRewireVoter(G, rho, alpha, gamma)
    Data = [Betti, SimplexCounts, Euler, Proportions]
    return 0

def Parallel_RewireTriangle(G, rhos, alphas, gammas):
    """
    """
    # Set rhos to be the set of initial densities of opinions
    # Set alphas to be the rewiring probability (social selection vs influence)
    # Set gammas to be the transitivity parameter 
    params = [[G], rhos, alphas]
    params = [p for p in product(*params)]

    # Get number of cpus that can be used
    num_cores = multiprocessing.cpu_count()

    # Actual call to parallel function
    results = Parallel(n_jobs=num_cores)(delayed(ParallelCall_RewireTriangle)(param) for param in tqdm(params))

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


# ============================================================================
# MAIN EXECUTION (Optional - for testing)
# ============================================================================
def main():
    """
    Main function for testing the module when run directly.
    This won't execute when the module is imported.
    """
    print("Testing module functions...")
    
    # Test your functions here
    test_data = [1, 2, 3, 4, 5]
    result = calculate_metrics(test_data, 'mean')
    print(f"Mean: {result}")
    
    processor = DataProcessor("test")
    print(processor)


if __name__ == "__main__":
    # This block only runs when the file is executed directly,
    # not when it's imported as a module
    main()