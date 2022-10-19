The repository contains code to analyze the anonymity of different network-level anonymity enhancing schemes for cryptocurrencies by using a bayesian framework and different heuristics.

## Different components

### Dandelion
* The folder contains three files to analyze anonymity when varying parameters such as:
    * Forwarding probability (pf)
    * Network size (N)
    * Fraction of adversary nodes in the network \(C\)

### Dandelion++
* This folder contains code for the same analysis as Dandelion i.e., varying pf, N and C. Along with that it contains code to:
    * Analyze anonymity when privacy subgraph is modified to be the bitcoin graph.
    * Construct privacy subgraphs by performing a graph learning attack. 

### LN (Lightning Network)
* This folder contains code to:
    * Construct a directed graph with corresponding nodes and edge weights from a real LN snapshot (consisting of channel announcement and policy updates).
    * Perform anonymity analysis when:
        * We select nodes as asversaries with varying:
            * Node degree
            * Centrality
        * Best K paths are considered for routing payment instead of a single best path.
        * The adversary is budget constrained and wants to select colluding nodes while maximizing deanonymization.
        * Random graph topologies with balanced centrality are used instead of the existing centralized real LN topologies.

## Setup Details
* Dependencies can be installed using requirements.txt file.
* The codes are tested to work for Python 3.6 or higher.
* The codes for Dandelion and Dandelion++ are self sufficient. However, to run the codes for LN, data from real LN network snapshots is required, which is provided in the form of JSON files.