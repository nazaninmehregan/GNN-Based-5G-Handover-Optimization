# TH-GCN: Graph-based Handover Management for 5G Vehicular Networks

## Overview
This thesis presents TH-GCN (THroughput-oriented Graph Convolutional Network), a novel approach to handover management in 5G vehicular networks. The research focuses on optimizing handover decisions in dense urban environments, addressing challenges such as frequent handovers, high mobility, and network load balancing.

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Scope and Assumptions](#problem-scope-and-assumptions)
3. [Methodology](#methodology)
4. [TH-GCN Algorithm](#th-gcn-algorithm)
5. [SINR-based Handover Decision Making](#sinr-based-handover-decision-making)
6. [TH-GCN Pipeline](#th-gcn-pipeline)
7. [Implementation](#implementation)
8. [Requirements](#requirements)
9. [Usage](#usage)

## Introduction
- **Context:** 5G technology for vehicular networks, focusing on low latency, high bandwidth, and high data rates
- **Challenges:** Low coverage range, frequent handovers (ping-pong effect), susceptibility to blockages, high mobility of vehicles
- **Handover Types:** Horizontal (intra-HO) and vertical (inter-HO); soft-HO and hard-HO

## Problem Scope and Assumptions
- **Environment:** Dense urban areas with segmented roads and varied vehicular speeds
- **Core Challenges:**
  - Load balancing on towers to prevent service degradation
  - Enhancing handover efficiency for high throughput and low latency
  - Designing a dynamic, adaptable solution for varied 5G tower deployments
- **Assumptions:**
  - Real-time measurement and reporting of network metrics
  - Capability to predictively model and mitigate handover effects
  - Advanced network technologies (e.g., 5G) deployed in the urban center

## Methodology
1. **Graph-based Network Modeling:**
   - Represent vehicles and towers as nodes
   - Edge features include throughput, RSSI, and distance
2. **GNN Architecture:**
   - Utilizes Graph Convolutional Network (GCN) layers
   - Incorporates edge features in the message passing process
3. **Incremental Learning:**
   - Model parameters saved and loaded for continuous adaptation

## TH-GCN Algorithm
1. **Graph Construction:** Create a graph representation of the network state
2. **Feature Extraction:** Use GCN to learn spatial features of the network
3. **Triplet Loss Training:** Optimize the model to distinguish between connected and non-connected towers
4. **Candidate Selection:** Identify top 3 tower candidates for each vehicle based on learned embeddings

TH-GCN Pipeline
The TH-GCN pipeline integrates data collection, graph construction, and GNN-based decision making into a cohesive system for handover management. The following figure illustrates the key components and flow of the TH-GCN pipeline:
Show Image
Figure 1: TH-GCN Pipeline Overview
Key components of the pipeline:

Data Collection: Gathering real-time network metrics from vehicles and towers.
Graph Construction: Creating a graph representation of the network state.
GNN Model: Applying the trained GNN to extract spatial features and generate embeddings.
Candidate Selection: Identifying top tower candidates based on learned embeddings.
SINR-based Decision: Making final handover decisions using SINR comparisons and hysteresis.

This pipeline demonstrates the integration of graph-based machine learning techniques with traditional network metrics for optimized handover management in 5G vehicular networks.

## SINR-based Handover Decision Making
1. **Data Collection:** Gather real-time network metrics (signal quality, throughput, distance, etc.)
2. **Candidate Evaluation:** Use TH-GCN to identify top tower candidates
3. **SINR Comparison:** Apply hysteresis-based algorithm for final handover decisions
4. **Handover Execution:** Perform handover or maintain connection based on decision criteria


# TH-GCN: Graph-based Handover Management for 5G Vehicular Networks

## Overview
This thesis presents TH-GCN (Tower Handover Graph Convolutional Network), a novel approach to handover management in 5G vehicular networks. The research focuses on optimizing handover decisions in dense urban environments, addressing challenges such as frequent handovers, high mobility, and network load balancing.


## TH-GCN Pipeline

The TH-GCN pipeline integrates data collection, graph construction, and GNN-based decision making into a cohesive system for handover management. The following figure illustrates the key components and flow of the TH-GCN pipeline:

![TH-GCN Pipeline](figures/pipeline.png)

*Figure 1: TH-GCN Pipeline Overview*

Key components of the pipeline:
1. **Data Collection**: Gathering real-time network metrics from vehicles and towers.
2. **Graph Construction**: Creating a graph representation of the network state.
3. **GNN Model**: Applying the trained GNN to extract spatial features and generate embeddings.
4. **Candidate Selection**: Identifying top tower candidates based on learned embeddings.
5. **SINR-based Decision**: Making final handover decisions using SINR comparisons and hysteresis.

This pipeline demonstrates the integration of graph-based machine learning techniques with traditional network metrics for optimized handover management in 5G vehicular networks.

[Remaining sections (Implementation, Requirements, Usage, Output) remain unchanged]


## Implementation
The core algorithm is implemented in Python, utilizing PyTorch and PyTorch Geometric for the GNN components.

### Key Components:
- `GNNWithEdgeFeatures`: Custom GNN model incorporating edge features
- `create_graph_data_for_timestamp`: Function to construct graph data from CSV input
- `triplet_loss`: Custom loss function for model training
- `get_triplets`: Function to generate triplets for training

## Requirements
- Python 3.x
- PyTorch
- PyTorch Geometric
- pandas
- numpy
- matplotlib
- networkx
- scikit-learn

## Usage
1. Ensure all required libraries are installed.
2. Place the `simulator_data.csv` file in the same directory as the script.
3. Run the script with the simulation timestamp as an argument:
   ```
   python <script_name>.py <simulation_timestamp>
   ```
4. The script will:
   - Load or initialize the GNN model
   - Train the model using the provided data
   - Generate embeddings for vehicles and towers
   - Identify top 3 tower candidates for each vehicle
   - Write results to `outputGNN.txt`

## Output
The `outputGNN.txt` file will contain the top 3 tower candidates for each vehicle, formatted as:
```
<vehicle_id>: [<tower_index_1>, <tower_index_2>, <tower_index_3>]
```

This implementation demonstrates the potential of graph-based approaches for optimizing handover management in 5G vehicular networks, addressing key challenges in urban environments.