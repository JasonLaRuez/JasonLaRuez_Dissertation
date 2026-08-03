"""
Module Name: Multilayer_ZZPH.py
Description: Contains functions for applying multilayer
zigzag persistence to a toy example and an EEG data set

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

import dionysus as d # C++ package with python bindings for persistent homology
import mne # Toolbox for EEG data
from mne_connectivity import spectral_connectivity_time # EEG connectivity measures (wPLI)

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

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

def ZZPH(Edges):
    """
    Given a sequence of edge sets, form a sequence of graphs,
    then for each graph construct the corresponding clique complex.
    Then, ZZPH is computed on this sequence of complexes and the 
    resultant persistence diagrams are returned.
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
    simplices = [list(key) for key in Times]; times = [Times[key] for key in Times]

    # Construct filtration and compute homology
    f = d.Filtration(simplices)
    zz, dgms, cells = d.zigzag_homology_persistence(f, times)

    return dgms

def Vectorization(D):
    """
    Takes in a symmetric matrix D, and returns a 1D vector containing
    all strictly upper triangular entries of D (excluding the diagonal).
    """
    # k=1 extracts entries above the main diagonal.
    # Change to k=0 if you wish to include the diagonal.
    return D[np.triu_indices_from(D, k=1)]

def MDS_Projection(vector_list, d=2, random_state=42):
    """
    Takes in a list of vectors (e.g., vectorized correlation matrices)
    and applies Multidimensional Scaling (MDS) to project them into d dimensions.

    Parameters:
    - vector_list: List or array of shape (n_samples, n_features)
    - d: Target dimension (default 2)
    - random_state: Seed for reproducibility

    Returns:
    - embedding: Array of shape (n_samples, d)
    """
    # Convert list to numpy array if it isn't one already
    X = np.array(vector_list)

    # Initialize MDS
    mds = MDS(n_components=d, random_state=random_state, normalized_stress='auto')

    # Fit and transform
    embedding = mds.fit_transform(X)

    return embedding

def PCA_Projection(vector_list, d=2, random_state=42):
    """
    Takes in a list of vectors and applies Principal Component Analysis (PCA)
    to reduce the dimension of the vectors to a given dimension d.

    Parameters:
    - vector_list: List or array of shape (n_samples, n_features)
    - d: Target dimension (number of components)
    - random_state: Seed for reproducibility (optional, though PCA is deterministic
                    unless solver is randomized, setting it is good practice)

    Returns:
    - embedding: Array of shape (n_samples, d)
    """
    # Convert list to numpy array
    X = np.array(vector_list)

    # Initialize PCA
    pca = PCA(n_components=d, random_state=random_state)

    # Fit and transform
    embedding = pca.fit_transform(X)

    return embedding

def UMAP_Projection(vector_list, d=2, random_state=42):
    """
    Takes in a list of vectors and applies Uniform Manifold Approximation and Projection (UMAP)
    to reduce the dimension of the vectors to a given dimension d.

    Parameters:
    - vector_list: List or array of shape (n_samples, n_features)
    - d: Target dimension
    - random_state: Seed for reproducibility

    Returns:
    - embedding: Array of shape (n_samples, d)
    """
    # Convert list to numpy array
    X = np.array(vector_list)

    # Initialize UMAP
    reducer = umap.UMAP(n_components=d, random_state=random_state)

    # Fit and transform
    embedding = reducer.fit_transform(X)

    return embedding

def tSNE_Projection(vector_list, d=2, random_state=42):
    """
    Takes in a list of vectors and applies t-Distributed Stochastic Neighbor Embedding (t-SNE)
    to reduce the dimension of the vectors to a given dimension d (typically 2).

    Parameters:
    - vector_list: List or array of shape (n_samples, n_features)
    - d: Target dimension (default 2)
    - random_state: Seed for reproducibility

    Returns:
    - embedding: Array of shape (n_samples, d)
    """
    # Convert list to numpy array
    X = np.array(vector_list)

    # Initialize t-SNE
    # 'init' parameter often set to 'pca' for better initialization
    perplexity = min(30, int(np.sqrt(len(X))))
    tsne = TSNE(n_components=d, random_state=random_state, init='pca', learning_rate='auto', perplexity=perplexity)

    # Fit and transform
    embedding = tsne.fit_transform(X)

    return embedding

def KruskalStress(D0, DK):
    """
    Computes the Kruskal stress (strain) between two distance matrices D0 and DK.
    Formula: sum((D0[i,j] - DK[i,j])**2) / sum(DK[i,j]**2) for i < j.
    """
    # Extract strictly upper triangular entries
    v0 = D0[np.triu_indices_from(D0, k=1)]
    vK = DK[np.triu_indices_from(DK, k=1)]

    numerator = np.sum((v0 - vK) ** 2)
    denominator = np.sum(vK ** 2)

    if denominator == 0:
        return 0.0 # Avoid division by zero

    return np.sqrt( numerator / denominator )

def Euclidean_D(embedding):
    """
    Takes in an embedding (n_samples x n_features) and computes the
    pairwise Euclidean distance matrix between each point.
    """
    return euclidean_distances(embedding)

def FNN(DK, DKp1, RTOL):
    """
    Computes the proportion of False Nearest Neighbors.

    Parameters:
    DK   : Symmetric distance matrix in dimension K
    DKp1 : Symmetric distance matrix in dimension K+1
    RTOL : Tolerance threshold
    """
    N = DK.shape[0]
    false_neighbors = 0
    total_checked = 0

    # Iterate over rows i
    for i in range(N):
        # We need to find j > i that minimizes DK[i, j]
        # Extract the row segment for j > i
        # Indices j > i are range(i+1, N)
        valid_j = np.arange(i+1, N)

        if len(valid_j) == 0:
            continue

        # Get distances for these j
        distances = DK[i, valid_j]

        # Find index of minimum distance within this segment
        min_idx_local = np.argmin(distances)
        j = valid_j[min_idx_local]

        # Distance in dimension K
        R_d = DK[i, j]

        # Distance in dimension K+1
        R_dp1 = DKp1[i, j]

        # Avoid division by zero
        if R_d == 0:
            # If points are identical in Dim K, check if they separated in K+1
            if R_dp1 > RTOL:
                 false_neighbors += 1
        else:
             # Standard FNN criterion calculation
             metric = np.sqrt(abs(R_dp1 - R_d) / R_d)
             if metric > RTOL:
                 false_neighbors += 1

        total_checked += 1

    if total_checked == 0:
        return 0.0

    return false_neighbors / total_checked


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

# ============================================================================
# EEG Data Code
# ============================================================================

def CollectPatientFiles(base_path):
    """
    Iterates through data directory (located at base_path)
    and collects the seizure files for each patient in a
    dictionary, and the nonseizure files for each patient in a
    separate dictionary.
    """
    # 24 patient files, though 1 and 22 are same person, 1.5 years apart
    patients = list(range(1, 25))

    # Initialize dictionaries to store file paths to EEG files with and without seizures
    patient_nonseizure_files = {}
    patient_seizure_files = {}

    for patient in patients:
        # Converts e.g. 1 to '01'
        patient_id = f'{patient:02d}'
        # Construct directory path for the patient
        dir_path = f'{base_path}/chb{patient_id}'

        if os.path.exists(dir_path):
            # List all .edf files in the directory that match the pattern
            files = [f for f in os.listdir(dir_path) if f.endswith('.edf') and f.startswith(f'chb{patient_id}')]
            files.sort() # Ensure they are in order

            # Store full paths
            full_paths = [os.path.join(dir_path, f) for f in files]
            seizure_paths, nonseizure_paths = [], []
            for path in full_paths:
                # For each edf file containing a seizure, there is an accompanying
                # file of the same name with .seizures after
                if os.path.isfile(path+'.seizures'):
                    seizure_paths.append(path)
                else:
                    nonseizure_paths.append(path)
            patient_seizure_files[patient] = seizure_paths
            patient_nonseizure_files[patient] = nonseizure_paths
            print(f"Patient {patient}: Found {len(full_paths)} files:")
            print(f"{len(seizure_paths)} seizure files and {len(nonseizure_paths)} non-seizure files.")
        else:
            print(f"Patient {patient}: Directory not found ({dir_path})")

    return patient_seizure_files, patient_nonseizure_files

def PatientSeizureTimes():
    """
    Returns the manually input seizure times for each patient and for
    each seizure file, as a nested dictionary with patients as keys,
    containing values as dictionaries with file numbers as keys
    and lists of time intervals indicating the [start, end] times of
    seizures (in seconds).
    """

    # keys are patients, then each patient has a dictionary where keys are files and values
    # correspond to intervals (s) during which a seizure occurred.

    patient_seizure_times = {
        '1': {
            '03': [[2996, 3036]],
            '04': [[1467, 1494]],
            '15': [[1732, 1772]],
            '10': [[1015, 1066]],
            '16': [[1015, 1066]],
            '18': [[1720, 1810]],
            '21': [[327, 420]],
            '26': [[1862, 1963]]},
        '2': {
            '16': [[130, 212]],
            '16+': [[2972, 3053]],
            '19': [[3369, 3378]]},
        '3': {
            '01': [[362, 414]],
            '02': [[731, 796]],
            '03': [[432, 501]],
            '04': [[2162, 2214]],
            '34': [[1982, 2029]],
            '35': [[2592, 2656]],
            '36': [[1725, 1778]]},
        '4': {
            '05': [[7804, 7853]],
            '08': [[6446, 6557]],
            '28': [[1679, 1781], [3782, 3898]]},
        '5': {
            '06': [[417, 532]],
            '13': [[1086, 1196]],
            '16': [[2317, 2413]],
            '17': [[2451, 2571]],
            '22': [[2348, 2465]]},
        '6': {
            '01': [[1724, 1738], [7461, 7476], [13525, 13540]],
            '04': [[327, 347], [6211, 6231]],
            '09': [[12500, 12516]],
            '10': [[10833, 10845]],
            '13': [[506, 519]],
            '18': [[7799, 7811]],
            '24': [[9387, 9403]]},
        '7': {
            '12': [[4920, 5006]],
            '13': [[3285, 3381]],
            '19': [[13688, 13831]]},
        '8': {
            '02': [[2670, 2841]],
            '05': [[2856, 3046]],
            '11': [[2988, 3122]],
            '13': [[2417, 2577]],
            '21': [[2083, 2347]]},
        '9': {
            '06': [[12231, 12295]],
            '08': [[2951, 3030], [9196, 9267]],
            '19': [[5299, 5361]]},
        '10': {
            '12': [[6313, 6348]],
            '20': [[6888, 6958]],
            '27': [[2382, 2447]],
            '30': [[3021, 3079]],
            '31': [[3801, 3877]],
            '38': [[4618, 4707]],
            '89': [[1383, 1437]]},
        '11': {
            '82': [[298, 320]],
            '92': [[2695, 2727]],
            '99': [[1454, 2206]]},
        '12': {
            '06': [[1665, 1726], [3415, 3447]],
            '08': [[1426, 1439], [1591, 1614], [1957, 1977], [2798, 2824], [3082, 3114], [3503, 3535]],
            '09': [[3082, 3114], [3503, 3535]],
            '10': [[593, 625], [811, 856]],
            '11': [[1085, 1122]],
            '23': [[253, 333], [425, 522], [630, 670]],
            '27': [[916, 951], [1097, 1124], [1728, 1753], [1921, 1963], [2388, 2440], [2621, 2669]],
            '28': [[181, 215]],
            '29': [[107, 146], [554, 592], [1163, 1199], [1401, 1447], [1884, 1921], [3557, 3584]],
            '33': [[2185, 2206], [2427, 2450]],
            '36': [[653, 680]],
            '38': [[1548, 1573], [2798, 2821], [2966, 3009], [3146, 3201], [3364, 3410]],
            '42': [[699, 750], [945, 973], [1170, 1199], [1676, 1701], [2213, 2236]]},
        '13': {
            '19': [[2077, 2121]],
            '21': [[934, 1004]],
            '40': [[142, 173], [530, 594]],
            '55': [[458, 478], [2436, 2454]],
            '58': [[2474, 2491]],
            '59': [[3339, 3401]],
            '60': [[638, 660]],
            '62': [[851, 916], [1626, 1691], [2664, 2721]]},
        '14': {
            '03': [[1986, 2000]],
            '04': [[1372, 1392], [2817, 2839]],
            '06': [[1911, 1925]],
            '11': [[1838, 1879]],
            '17': [[3239, 3259]],
            '18': [[1039, 1061]],
            '27': [[2833, 2849]]},
        '15': {
            '06': [[272, 397]],
            '10': [[1082, 1113]],
            '15': [[1591, 1748]],
            '17': [[1925, 1960]],
            '20': [[607, 662]],
            '22': [[760, 965]],
            '28': [[876, 1066]],
            '31': [[1751, 1871]],
            '40': [[834, 894], [2378, 2497], [3362, 3425]],
            '46': [[3322, 3429]],
            '49': [[1108, 1248]],
            '52': [[778, 849]],
            '54': [[263, 318], [843, 1020], [1524, 1595], [2179, 2250], [3428, 3460]],
            '62': [[751, 859]]},
        '16': {
            '10': [[2290, 2299]],
            '11': [[1120, 1129]],
            '14': [[1854, 1868]],
            '16': [[1214, 1220]],
            '17': [[227, 236], [1694, 1700], [2162, 2170], [3290, 3298]],
            '18': [[627, 635], [1909, 1916]]},
        '17': {
            '03': [[2282, 2372]],
            '04': [[3025, 3140]],
            '63': [[3136, 3224]]},
        '18': {
            '29': [[3477, 3527]],
            '30': [[541, 571]],
            '31': [[2087, 2155]],
            '32': [[1908, 1963]],
            '35': [[2196, 2264]],
            '36': [[463, 509]]},
        '19': {
            '28': [[299, 377]],
            '29': [[2964, 3041]],
            '30': [[3159, 3240]]},
        '20': {
            '12': [[94, 123]],
            '13': [[1440, 1470], [2498, 2537]],
            '14': [[1971, 2009]], 
            '15': [[390, 425], [1689, 1738]],
            '16': [[2226, 2261]],
            '68': [[1393, 1432]]},
        '21': {
            '19': [[1288, 1344]],
            '20': [[2627, 2677]],
            '21': [[2003, 2084]],
            '22': [[2553, 2565]]},
        '22': {
            '20': [[3367, 3425]],
            '25': [[3139, 3213]],
            '38': [[1263, 1335]]},
        '23': {
            '06': [[3962, 4075]],
            '08': [[325, 345], [5104, 5151]],
            '09': [[2589, 2660], [6885, 6947], [8505, 8532], [9580, 9664]]},
        '24': {
            '01': [[480, 505], [2451, 2476]],
            '03': [[231, 260], [2883, 2908]],
            '04': [[1088, 1120], [1411, 1438], [1745, 1764]],
            '06': [[1229, 1253]],
            '07': [[38, 60]],
            '09': [[1745, 1764]],
            '11': [[3527, 3597]],
            '13': [[3288, 3304]],
            '14': [[1939, 1966]],
            '15': [[3552, 3569]],
            '17': [[3515, 3581]],
            '21': [[2804, 2872]]}
    }
    return patient_seizure_times

def PlotSeizureTimes(patient_seizure_times):
    """
    Plots the distribution of all seizure lengths in the data set, then
    plots (as points) the mean seizure length for each patient.
    """

    all_lengths = []
    patient_means = []
    patient_ids = sorted(patient_seizure_times.keys(), key=lambda x: int(x))

    for pid in patient_ids:
        lengths = []

        for file_intervals in patient_seizure_times[pid].values():
            for start, end in file_intervals:
                length = end - start
                lengths.append(length)
                all_lengths.append(length)

        patient_means.append(np.mean(lengths))

    # Plot 
    plt.figure(figsize=(10, 5))
    plt.boxplot(all_lengths, widths=0.4, showfliers=False, orientation='horizontal')
    y_pos = np.ones(len(patient_means))
    plt.scatter(patient_means, y_pos, zorder=3, label='Patient Means')
    plt.xlabel("Seizure length (seconds)")
    plt.title("Distribution of All Seizure Times with Per-Patient Means")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Five Number Summary
    minimum = np.min(all_lengths)
    q1 = np.percentile(all_lengths, 25)
    median = np.median(all_lengths)
    q3 = np.percentile(all_lengths, 75)
    maximum = np.max(all_lengths)

    stats_text = (
        f"Min: {minimum:.1f}\n"
        f"Q1: {q1:.1f}\n"
        f"Median: {median:.1f}\n"
        f"Q3: {q3:.1f}\n"
        f"Max: {maximum:.1f}"
    )

    # Place text in top right (using relative coordinates)
    # transform=plt.gca().transAxes makes (0,0) bottom-left and (1,1) top-right of the axes
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def clean_channel_name(name):
    """
    Deals with duplicate files. If two channels are duplicates, mne adds an
    integer to the end of the name. This function removes the integer if it is
    0 so we can rename the 0 and remove the other duplicates
    """
    if name.count('-') == 2 and name.rsplit('-',1)[1] == '0':
        return name.rsplit('-', 1)[0]
    return name

def Preprocessing(file_path):
    """
    For a given file_path, (i) data is loaded, (ii) all channels besides the common
    18 reference channels are discarded, (iii) a high-pass filter (1 Hz) is
    applied, removing drift (and destroying signals below 1 Hz), (iv) a notch
    filter is applied at 60 Hz to remove power-line frequency and harmonics
    (60 Hz is standard in U.S. electrical systems). Then the processed
    data is returned.
    """

    Channels = ['FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2','FZ-CZ', 'CZ-PZ']      
    l_freq = 1.0            # high-pass cutoff
    line_freqs = [60]       # based on PSD

    # Load data
    raw = mne.io.read_raw_edf(file_path, preload=True) 

    channels = raw.ch_names
    chanels_to_drop = []

    for channel in channels:
        cleaned_channel = clean_channel_name(channel)
        if cleaned_channel not in Channels:
            chanels_to_drop.append(channel)
        elif cleaned_channel != channel:
            raw.rename_channels({channel: cleaned_channel})

    raw.drop_channels(chanels_to_drop)
    channels = raw.ch_names

    if set(channels) != set(Channels):
        print(f"Channel mismatch in file {file_path}")
        return -1

    # Ensure EEG channel type is set
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    # High-pass filter: 1 Hz, zero-phase FIR (PREP-style)
    raw.filter(
        l_freq=l_freq,
        h_freq=None,
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",)

    # Notch filter at power-line frequency and harmonic
    raw.notch_filter(
        freqs=line_freqs,
        method="fir",        # stable and zero-phase in forward/backward
        phase="zero",)

    return raw

def GetWindowedData(raw, channel, window_idx, width = 6, shift = 1, sfreq = 256):
    """
    width: window width in seconds
    shift: window shift in seconds
    channel: channel name or index
    window_idx: window index
    Returns the windowed time series for the given channel
    """
    # Start and end times in seconds
    start = window_idx * shift
    end = start + width
    # Index data using the sampling frequency
    start_stop_seconds = np.array([start, end])
    start_sample, stop_sample = (start_stop_seconds * sfreq).astype(int)
    windowed_data = raw[channel, start_sample:stop_sample]

    return windowed_data

def GetCorrelationMatrix(raw, channels, window_idx, width = 6, shift = 1, sfreq = 256):
    """
    For each channel, and for the given time window, obtains a windowing of each channel
    time series, then computes the pairwise correlation matrix for all channels.
    """
    windowed_data = dict()
    for channel_name in channels:
        # Get the integer index for the channel name
        channel_index = channels.index(channel_name)
        windowed_data[channel_name] = GetWindowedData(raw, channel_index, window_idx, width, shift, sfreq)

    num_channels = len(channels)
    correlation_matrix = np.identity(num_channels) # Initialize with 1s on the diagonal

    # Compute correlations for all unique pairs
    for i, ch1_name in enumerate(channels):
        for j, ch2_name in enumerate(channels):
            if i < j: # Avoid redundant calculations and self-correlation
                series1 = windowed_data[ch1_name][0][0] # Access the actual time series (first element of tuple, then first element of array)
                series2 = windowed_data[ch2_name][0][0] # Access the actual time series

                # Ensure series are not empty and have valid data
                if series1.size > 1 and series2.size > 1:
                    # Compute Pearson correlation coefficient
                    correlation = np.corrcoef(series1, series2)[0, 1]
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
                else:
                    # Handle cases with insufficient data for correlation
                    correlation_matrix[i, j] = np.nan # Or 0, depending on desired behavior
                    correlation_matrix[j, i] = np.nan

    return correlation_matrix

def GetCorrelationMatrices(raw, channels, window_indices, width, shift, sfreq=256):
    """
    raw: MNE Raw object
    channels: List of channel names
    width: window width in seconds
    shift: window shift in seconds
    window_indices: List of window indices
    sfreq: Sampling frequency
    Returns an array of correlation matrices (tensor) for each window index,
    representing the temporal evolution of the correlation matrix between
    sliding windows of EEG channel time series data.
    """
    correlation_matrices = []
    for window_idx in window_indices:
        matrix = GetCorrelationMatrix(raw, channels, window_idx, width, shift, sfreq)
        correlation_matrices.append(matrix)
    return np.array(correlation_matrices)

def CorrelationNetwork_threshold(correlation_matrices, threshold):
    """
    For a sequence of correlation matrices, returns an edge set corresponding
    to a thresholded correlation network where edges are formed between nodes
    whose pearson correlation has magnitude greater than threshold.
    """
    Edges = [set() for _ in range(len(correlation_matrices))]

    for step, matrix in enumerate(correlation_matrices):
        for i in range(matrix.shape[0]):
            for j in range(i+1, matrix.shape[0]):
                if abs(matrix[i, j]) > threshold:
                    Edges[step].add(tuple([i, j]))
    return Edges

def CorrelationNetwork_density(correlation_matrices, density):
    """
    For a sequence of correlation matrices, returns an edge set corresponding
    to a thresholded correlation network where edges are formed between nodes
    with the highest magnitude correlations until the edge density of the
    resulting graph has exceeded the given density.
    """
    Edges = [set() for _ in range(len(correlation_matrices))]

    for step, matrix in enumerate(correlation_matrices):
        N = matrix.shape[0]
        # Get indices for the upper triangle (excluding diagonal)
        rows, cols = np.triu_indices(N, k=1)

        # Get absolute correlation values
        vals = np.abs(matrix[rows, cols])

        # Sort indices by value in descending order
        sorted_indices_idx = np.argsort(vals)[::-1]

        # Calculate number of edges needed to exceed density
        # Density = num_edges / total_possible_edges
        total_possible = math.comb(N, 2)
        num_edges = int(density * total_possible) + 1

        # Add the top edges
        for k in range(min(num_edges, total_possible)):
            idx = sorted_indices_idx[k]
            i, j = rows[idx], cols[idx]
            Edges[step].add(tuple(sorted([i, j])))

    return Edges

def CorrelationNetwork_KNN(correlation_matrices, K):
    """
    For a sequence of correlation matrices, returns an edge set corresponding
    to a KNN correlation network where for each node v, edges are added to
    other nodes with the K highest absolute correlations with node v.
    """
    Edges = [set() for _ in range(len(correlation_matrices))]

    for step, matrix in enumerate(correlation_matrices):
        for i in range(matrix.shape[0]):
            # Calculate absolute correlations for node i
            corrs = np.abs(matrix[i, :])
            # Set self-correlation to -1 to ensure it is not selected
            corrs[i] = -1.0
            # Get indices of the K highest correlations
            neighbors = np.argsort(corrs)[-K:]

            for j in neighbors:
                Edges[step].add(tuple(sorted([i, j])))

    return Edges


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