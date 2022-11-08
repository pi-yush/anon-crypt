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

### LN (Lightning Network)
* This folder contains code to perform analysis for:
    * An older 2018 LN topology snapshot
    * The most recent 2021 LN snapshot
    * Longitudinal analysis of topologies from 2019 and 2020
    * Random graph topologies with balanced centrality are used instead of the existing centralized real LN topologies.
* All the folders consists of three code files:
    * Construct a directed graph with corresponding nodes and edge weights from a real LN snapshot (consisting of channel announcement and policy updates).
    * Perform anonymity analysis when we select nodes as asversaries with varying node degree.
    * Code to plot the data generated by the analysis scripts.
* There is also a seperate file for analysis when the best K paths are considered for routing payment instead of a single best path.       
* The documentation about working of the code has been done for the 2018 topo files (other folders also have similar codes with minor differences).

## Setup Details
* Dependencies can be installed using requirements.txt file.
* The codes are tested to work for Python 3.6 or higher.
* The codes for Dandelion and Dandelion++ are self sufficient. However, to run the codes for LN, data from real LN network snapshots is required, which is provided in the form of JSON files (see releases).
* LN code outputs data in different directories. This data can  be used to perform analysis of plot data (using the plotting scripts provided in each folder)
