import networkx as nx
import pandas as pd
import numpy as np
import sys
import torch
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, ChebConv
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch_geometric.data import Data
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# test_start_time = float(sys.argv[1])
# train_end_time = test_start_time - 0.5


dataset_path = 'simulator_data_1.csv'

# Load the dataset
# dataset_path = "path_to_your_dataset.csv"  # Replace with your actual dataset path
data = pd.read_csv(dataset_path)

# Function to compute the distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# ------------------------------------------------------------------

# Function to create a graph data object and NetworkX graph for a given timestamp
# Modify your function to process each timestamp as follows:
def create_graph_data_for_timestamp(df, timestamp,  vehicle_ids, tower_ids):
    timestamp_data = df[df['timestamp'] == timestamp]


    # Map specific vehicle and tower IDs to consecutive indices
    vehicle_mapping = {vid: i for i, vid in enumerate(vehicle_ids)}
    tower_mapping = {tid: i + len(vehicle_ids) for i, tid in enumerate(tower_ids)}

    node_features = []
    edge_index = []
    edge_features = []

    # Process vehicle nodes
    for vid in vehicle_ids:
        vehicle_data = timestamp_data[timestamp_data['vehicleId'] == vid]
        if not vehicle_data.empty:
            avg_speed = vehicle_data['vehicleSpeed'].mean()
            avg_dir = vehicle_data['vehicleDirection'].mean()
            avg_pos_x = vehicle_data['vehiclePosX'].mean()
            avg_pos_y = vehicle_data['vehiclePosY'].mean()
            vehicle_features = [avg_speed, avg_dir, avg_pos_x, avg_pos_y]
        else:
            # Provide some default or mean features if the vehicle is not present
            vehicle_features = [0, 0, 0, 0]  # Placeholder, adjust as needed
        node_features.append(vehicle_features)

    # Process tower nodes
    for tid in tower_ids:
        tower_data = timestamp_data[timestamp_data['masterId'] == tid]
        if not tower_data.empty:
            avg_pos_x = tower_data['masterPosX'].mean()
            avg_pos_y = tower_data['masterPosY'].mean()
            avg_load = tower_data['masterLoad'].mean()
        else:
            # Default or mean features for absent towers
            avg_pos_x = avg_pos_y = avg_load = 0  # Placeholder
        tower_features = [avg_pos_x, avg_pos_y, avg_load, 0.0]  # Last value is placeholder
        node_features.append(tower_features)

    # Add edges for existing pairs in this timestamp
    for _, row in timestamp_data.iterrows():
        if row['vehicleId'] in vehicle_mapping and row['masterId'] in tower_mapping:
            vehicle_id = vehicle_mapping[row['vehicleId']]
            tower_id = tower_mapping[row['masterId']]
            edge_index.append([vehicle_id, tower_id])
            edge_features.append([row['throughput'], row['masterRssi'], row['distance']])

    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_features_tensor)

# -----------------------------------------------------------------------------------------

#gets sequence as input and predicts the next step of sequence using GCN
#
class TemporalGNN(nn.Module):
    def __init__(self, node_features_dim, edge_features_dim, memory_dim):
        super(TemporalGNN, self).__init__()
        self.conv = GCNConv(node_features_dim, memory_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * memory_dim, 2 * memory_dim),
            nn.ReLU(),
            nn.Linear(2 * memory_dim, edge_features_dim)
        )

    def forward(self, x, edge_index):
        node_embeddings = self.conv(x, edge_index)
        src, dest = edge_index
        edge_features = torch.cat([node_embeddings[src], node_embeddings[dest]], dim=1)
        predicted_edge_features = self.edge_predictor(edge_features)
        return predicted_edge_features

#predictive loss function because we don't have labels
#we have to define a criteria for loss/ whether my model is working and validates
#without labels we have to self-supervise
#our criteria is common edge
def predictive_loss(predicted_edge_features, predicted_edge_index, actual_edge_features, actual_edge_index):
    # Make sure tuple is still a Python built-in function here
    # print(type(tuple))  # Debug: This should output <class 'type'>

    predicted_edges = set(map(tuple, predicted_edge_index.t().tolist()))
    actual_edges = set(map(tuple, actual_edge_index.t().tolist()))

    # Find common edges
    common_edges = predicted_edges.intersection(actual_edges)
    # print("common edges:", len(common_edges), " / predicted: ", len(predicted_edges), " / actual: ", len(actual_edges))

    # Filter out the features for the common edges
    pred_features = []
    actual_features = []
    for i, edge in enumerate(predicted_edge_index.t().tolist()):
        if tuple(edge) in common_edges:
            pred_features.append(predicted_edge_features[i])
    for i, edge in enumerate(actual_edge_index.t().tolist()):
        if tuple(edge) in common_edges:
            actual_features.append(actual_edge_features[i])

    # Check if lists are empty, and handle this case
    if not pred_features or not actual_features:
        return torch.tensor(0.0, requires_grad=True)

    # Calculate MSE Loss on common edges
    pred_features = torch.stack(pred_features)
    actual_features = torch.stack(actual_features)
    return torch.nn.functional.mse_loss(pred_features, actual_features)


# ------------------------------------------------------------------------------------------

# import torch
df = data
# Convert timestamps to numeric and sort
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
df.sort_values('timestamp', inplace=True)

train_end_time = 8  # Define the timestamp up to which data should be used for training
test_start_time = 8.5 

# Select train and test data based on timestamps
train_data = df[df['timestamp'] <= train_end_time]
test_data = df[df['timestamp'] == test_start_time]

# Example of creating graph data for each set
vehicle_ids = sorted(df['vehicleId'].unique())  # Ensure you have a list of all vehicle IDs
tower_ids = sorted(df['towerId'].unique())  # Ensure you have a list of all tower IDs

# Function to create graph data (assuming it's already defined)
train_graphs = [create_graph_data_for_timestamp(train_data, ts, vehicle_ids, tower_ids) for ts in train_data['timestamp'].unique()]
test_graphs = [create_graph_data_for_timestamp(test_data, ts, vehicle_ids, tower_ids) for ts in test_data['timestamp'].unique()]


model = TemporalGNN(node_features_dim=4, edge_features_dim=3, memory_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epoch = 200
# Training loop
for epoch in range(num_epoch):  # Number of training epochs
    print(f"Starting Epoch {epoch+1}")
    for graph_data in train_graphs:
        optimizer.zero_grad()
        predicted_edge_features = model(graph_data.x, graph_data.edge_index)
        loss = predictive_loss(predicted_edge_features, graph_data.edge_index, graph_data.edge_attr, graph_data.edge_index)
        loss.backward()
        optimizer.step()
    print(f"Completed training for Epoch {epoch+1} ")



# -------------------------------------------------------------------------------

vehicle_mapping = {vid: i for i, vid in enumerate(vehicle_ids)}

# Create mapping from tower ID to a unique index
tower_mapping = {tid: i for i, tid in enumerate(tower_ids)}

def map_predictions_to_ids(predicted_edge_features, edge_index, vehicle_mapping, tower_mapping):
    mapped_predictions = {}
    vehicle_ids = list(vehicle_mapping.keys())  # Ensure these mappings are correct
    tower_ids = list(tower_mapping.keys())

    for i, edge in enumerate(edge_index.t().tolist()):
        vehicle_id = vehicle_ids[edge[0]]
        tower_id = tower_ids[edge[1] - len(vehicle_ids)]  # Ensure the index is correctly adjusted for towers
        mapped_predictions[(vehicle_id, tower_id)] = predicted_edge_features[i].tolist()

    return mapped_predictions


# Assuming you have your model predictions per graph, you'll need to process each graph individually
for graph_data in test_graphs:
    model.eval()
    with torch.no_grad():
        predicted_edge_features = model(graph_data.x, graph_data.edge_index)
        # print(f"Predicted edge features for timestamp {test_start_time}: {predicted_edge_features}")

    # Map predictions back to vehicle and tower IDs
    mapped_predictions = map_predictions_to_ids(predicted_edge_features, graph_data.edge_index, vehicle_mapping, tower_mapping)

    # Print the mapped predictions
    numberofpairs=0
    for pair, features in mapped_predictions.items():
        numberofpairs = numberofpairs+1
        print(f"Vehicle {pair[0]}__Tower {pair[1]} -> Throughput: {features[0]}, RSSI: {features[1]}, Distance: {features[2]}")

