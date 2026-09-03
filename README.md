# JasonLaRuez_Dissertation
Code for the PhD dissertation "Topological Tools for the Analysis of Complex Networks and Higher-Order Networks" (Jason LaRuez, RIT, 2026): persistent homology, Betti numbers, and simpliciality measures applied to temporal, evolving, and coevolving networks and hypergraphs, including multi-layer zigzag persistence and a generalized preferential-attachment hypergraph model.

## Related paper

This repository also contains the code for the paper ["Preferential Attachment as a Simpliciality-Enforcing Mechanism in Hypergraphs"](https://arxiv.org/pdf/2608.09788) by Jason LaRuez and Brendan Rooney. That paper's code lives in `Chapter2/HypergraphModels.py` and `Chapter2/HypergraphModels_Notebook.ipynb`.

## Repository structure

Each chapter folder contains one or more `.py` script / `.ipynb` notebook pairs. The notebook imports its script's functions (`from <Script> import *`) and drives the actual simulations, parameter sweeps, and figure generation; the script holds the reusable function definitions.

### Chapter1 — Multi-layer zigzag persistence

- **`Multilayer_ZZPH.py`** / **`Multilayer_ZZPH_Notebook.ipynb`**

Applies multilayer zigzag persistent homology to a synthetic periodic ring network: nodes arranged in a ring gain edges with a sinusoidally time-varying probability, and edges from adjacent time windows are aggregated into successive "layers." The notebook shows how this layer aggregation reveals a persistent ring structure ($H_1$) that is not visible from any single layer alone, and generates the dissertation's Figures 1.4 and 1.7–1.11.

### Chapter2 — Generative models, simpliciality, and LittleSIS

Three independent script/notebook pairs:

- **`HypergraphModels.py`** / **`HypergraphModels_Notebook.ipynb`** — code for the arXiv paper above. Generative hypergraph models (Erdos-Renyi, Watts-Strogatz — including simpliciality-enforcing variants, and both linear and nonlinear preferential attachment), simpliciality measures ($\sigma_{SF}$ simplicial fraction and $\sigma_{FES}$ face edit simpliciality), persistent homology, cycle rank, and filling efficiency ($\gamma_k$), plus the parallel parameter-sweep and plotting infrastructure used to produce the paper's figures.
- **`LittleSIS.py`** / **`LittleSIS_Data.ipynb`** — applies the same topological toolkit to real-world data from the [LittleSIS](https://littlesis.org) relationship database: builds a temporally-growing (and, separately, a zigzag-filtered add/remove) graph from relationship records, and tracks Betti numbers, cycle rank, and filling efficiency over time.
- **`NetworkModels.py`** / **`NetworkModels_Notebook.ipynb`** — graph-only analogues (Erdos-Renyi, Barabasi-Albert, nonlinear preferential attachment, Watts-Strogatz) of the hypergraph models above, used as a baseline for comparison.

### Chapter3 — Coevolving voter models

- **`VoterModels.py`** / **`VoterModels_Notebook.ipynb`**

Coevolving (adaptive-network) voter models on both graphs and hypergraphs, with three rewiring mechanisms (rewire-to-random, rewire-to-same, and triangle-closing rewire-to-same), governed by $\alpha$ (probability of structural rewiring vs. social influence) and, for the triangle-closing variant, $\gamma$ (probability of triangle-closure rewiring vs. plain rewire-to-same). Social influence is applied via majority or proportional voting. Tracks simpliciality ($\sigma_{SF}$, $\sigma_{ES}$), Betti numbers, cycle rank, and filling efficiency for the hypergraph variants, and includes figures comparing outcomes across the graph and hypergraph models and across initial network topologies (ER/BA/WS).

## Data

Simulation outputs and datasets are not included in this repository (the pickled results are many GB). File paths are passed in as function arguments (`base_path`, `BASE_G`, `proc_base`, etc.) rather than resolved through a shared configuration file, so running any given cell requires supplying paths to your own data directory.

## Dissertation

The full dissertation text is included as `Jason_Dissertation.pdf`.
