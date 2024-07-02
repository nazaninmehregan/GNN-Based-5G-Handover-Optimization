import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
import sys
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch
from sklearn.metrics.pairwise import cosine_similarity
import random

sim_timestamp = float(sys.argv[1])

# print(sim_timestamp)

filepath = '/home/nazanin/ProjectHOMng/simu5G/src/stack/phy/layer/'
dataset_path = filepath+'simulator_data.csv'


# Load the dataset
# dataset_path = "path_to_your_dataset.csv"  # Replace with your actual dataset path
data = pd.read_csv(dataset_path)

# Function to compute the distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to check for handovers between timestamps
def check_handovers(df, current_timestamp, next_timestamp):
    current_data = df[df['timestamp'] == current_timestamp]
    next_data = df[df['timestamp'] == next_timestamp]

    handovers = {}
    for vid in current_data['vehicleId'].unique():
        current_master = current_data[current_data['vehicleId'] == vid]['masterId'].iloc[0]
        next_master = next_data[next_data['vehicleId'] == vid]['masterId'].iloc[0] if vid in next_data['vehicleId'].values else current_master
        handovers[vid] = current_master != next_master

    return handovers

#---------------------------------------------------------------------------------------------#

# Function to create a graph data object and NetworkX graph for a given timestamp
def create_graph_data_for_timestamp(df, timestamp):
    timestamp_data = df[df['timestamp'] == timestamp]
    vehicle_ids = timestamp_data['vehicleId'].unique()
    # print('number of vehicles:', vehicle_ids.size)
    tower_ids = timestamp_data['towerId'].unique()
    # print('number of towers:', tower_ids.size)
    vehicle_mapping = {vid: i for i, vid in enumerate(vehicle_ids)}
    print(vehicle_mapping)
    tower_mapping = {tid: i + len(vehicle_ids) for i, tid in enumerate(tower_ids)}

    node_features = []
    edge_index = []
    edge_features = []

    G = nx.Graph()  # NetworkX Graph for visualization

    # Process vehicle nodes
    # there could be at most 3 rows and at least 1 row for each vehicle id in each timestamp
    # we want to make sure we get the latest data for each vehicle
    for vid in vehicle_ids:
        try:
            vehicle_data = timestamp_data[timestamp_data['vehicleId'] == vid].iloc[2]
        except IndexError:
            try:
                vehicle_data = timestamp_data[timestamp_data['vehicleId'] == vid].iloc[1]
            except IndexError:
                vehicle_data = timestamp_data[timestamp_data['vehicleId'] == vid].iloc[0]
        
        vehicle_features = [
            vehicle_data['vehicleSpeed'], vehicle_data['vehicleDirection'], vehicle_data['vehiclePosX'],
            vehicle_data['vehiclePosY']
        ]
        node_features.append(vehicle_features)
        G.add_node(vehicle_mapping[vid], speed=vehicle_data['vehicleSpeed'], dir=vehicle_data['vehicleDirection'], 
                  pos=(vehicle_data['vehiclePosX'], vehicle_data['vehiclePosY']),
                   type='vehicle', label=f'{vid}')

    # Process tower nodes
    for tid in tower_ids:
        master_data = timestamp_data[timestamp_data['masterId'] == tid][::-1]
        if not master_data.empty:
            tower_data = master_data.iloc[0]
            tower_features = [
                tower_data['masterPosX'], tower_data['masterPosY'],
                tower_data['masterLoad'], 0.0  # Placeholder values for missing features
            ]
            G.add_node(tower_mapping[tid], pos=(tower_data['masterPosX'], tower_data['masterPosY']), load=tower_data['masterLoad'],
                    type='tower', label=f'T{tid}')
        else:
            tower_data = timestamp_data[timestamp_data['towerId'] == tid][::-1].iloc[0]
            tower_features = [
                tower_data['towerPosX'], tower_data['towerPosY'],
                tower_data['towerLoad'], 0.0  # Placeholder values for missing features
            ]
            G.add_node(tower_mapping[tid], pos=(tower_data['towerPosX'], tower_data['towerPosY']), load=tower_data['towerLoad'],
                        type='tower', label=f'T{tid}')

        node_features.append(tower_features)
        

    # Add edges based on the masterId
    for _, row in timestamp_data.iterrows():
        vehicle_id = vehicle_mapping[row['vehicleId']]
        tower_id = tower_mapping[row['masterId']]  # Connect to the master tower
        edge_index.append([vehicle_id, tower_id])
        

        distance = calculate_distance(row['vehiclePosX'], row['vehiclePosY'],
                                      row['masterPosX'], row['masterPosY'])
        edge_features.append([row['throughput'], row['masterRssi'], distance])

        # Add edges to the NetworkX graph
        G.add_edge(vehicle_id, tower_id, throughput=row['throughput'],  rssi=row['masterRssi'], distance=distance)
    
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # print(edge_index_tensor)
    # print(len(edge_index))
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)

    graph_data = Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_features_tensor)
    # print("node features: ", graph_data.x.shape,"edge index: ", graph_data.edge_index.shape,"edge attr: ", graph_data.edge_attr.shape)
    return graph_data, G, vehicle_mapping, tower_mapping

#--------------------------------------------------------------------------------------------------#

# Two GCNConv layers for processing node features.
# A forward method to propagate node features through the network.

class GNN(torch.nn.Module):

    def __init__(self, num_node_features, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)  # Include edge_index here
        return x
    
#---------------------------------------------------------------------------------------------------#


# Example of preparing the data
graph_data, G, vehicle_mapping, tower_mapping = create_graph_data_for_timestamp(data, sim_timestamp)
# print(vehicle_mapping)
# print(tower_mapping)
# print(graph_data.num_node_features)

# Create a DataLoader to handle batching during training.
loader = DataLoader([graph_data], batch_size=32, shuffle=True)

#A custom triplet loss function is defined to ensure that the distance between anchor-positive pairs 
#is smaller than the distance between anchor-negative pairs by a margin.
#The triplet loss is computed as the mean of the losses for all triplets in a batch.

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

#------------------------------------------------------------------------------------------------#


# constructs triplets (anchor, positive, negative) for training the model based on vehicle-tower connections.

def get_triplets(data, embeddings, vehicle_mapping, tower_mapping):
    anchors = []
    positives = []
    negatives = []

    # print(data.edge_index)
    for vehicle_id in vehicle_mapping.values():
        
        connected_towers = data.edge_index[1][data.edge_index[0] == vehicle_id].tolist()
        # print('connected towers: ',connected_towers)

        if not connected_towers:
            continue

        # Select one connected tower as positive
        positive_idx = random.choice(connected_towers)

        # Select one non-connected tower as negative
        non_connected_towers = list(set(tower_mapping.values()) - set(connected_towers))
        if not non_connected_towers:
            continue
        negative_idx = random.choice(non_connected_towers)

        anchors.append(embeddings[vehicle_id])
        positives.append(embeddings[positive_idx])
        negatives.append(embeddings[negative_idx])

    # Convert to tensors
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    return anchors, positives, negatives
#----------------------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(num_node_features=graph_data.num_node_features, hidden_channels=64, out_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# The GNN model is instantiated and trained using the triplet loss function over multiple epochs.
# An optimizer (Adam) is used to update the model parameters.

# print('================TRAINING THE MODEL================')
for epoch in range(50):
    model.train()
    total_loss = 0
    for data in loader:

        data = data.to(device)
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)

        anchors, positives, negatives = get_triplets(data, embeddings, vehicle_mapping, tower_mapping)
        loss = triplet_loss(anchors, positives, negatives)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')

#-------------------------------------------------------------------------------------------------------#



vehicle_indices = list(vehicle_mapping.values())
tower_indices = list(tower_mapping.values())

# After training, vehicle and tower embeddings are extracted.
# Cosine similarity between vehicle and tower embeddings is calculated.
# Top 3 candidate towers for handovers are identified based on the similarity scores.
    
vehicle_embeddings = embeddings[vehicle_indices]
tower_embeddings = embeddings[tower_indices]

# Convert PyTorch tensors to NumPy arrays using detach().numpy()
# vehicle_embeddings = vehicle_embeddings.detach().numpy()
vehicle_embeddings = vehicle_embeddings.cpu().detach().numpy()

# tower_embeddings = tower_embeddings.detach().numpy()
tower_embeddings = tower_embeddings.cpu().detach().numpy()


# Calculate cosine similarity between each vehicle and all towers
similarity_matrix = cosine_similarity(vehicle_embeddings, tower_embeddings)

top_3_candidates = {}
for i, vehicle_id in enumerate(vehicle_mapping.keys()):
    # Get indices of top 3 towers based on similarity
    top_towers_indices = np.argsort(similarity_matrix[i])[::-1][:3]

    top_3_candidates[vehicle_id] = top_towers_indices
    # print(f'Vehicle {vehicle_id} top handover for next timestamp will be {top_towers_indices}')

with open(filepath+'outputGNN.txt', 'w+') as f:
    for vehicle_id, top_towers_indices in top_3_candidates.items():
        f.write(f'{vehicle_id}: {top_towers_indices}\n')

