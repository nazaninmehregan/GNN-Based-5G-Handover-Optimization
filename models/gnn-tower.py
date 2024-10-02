import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

sim_timestamp = float(sys.argv[1])

# filepath = 'sim_results/'
filepath = ''
dataset_path = filepath+'simulator_data.csv'
# dataset_path = 'simulator_data.csv'

# Load the dataset
data = pd.read_csv(dataset_path)

def create_graph_data_for_timestamp(df, timestamp):
    timestamp_data = df[df['timestamp'] == timestamp]
    vehicle_ids = timestamp_data['vehicleId'].unique()
    tower_ids = timestamp_data['towerId'].unique()
    vehicle_mapping = {vid: i for i, vid in enumerate(vehicle_ids)}
    tower_mapping = {tid: i + len(vehicle_ids) for i, tid in enumerate(tower_ids)}

    node_features = []
    edge_index = []
    edge_features = []

    G = nx.Graph()

    # Process vehicle nodes
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
                tower_data['masterLoad'], 0.0
            ]
            G.add_node(tower_mapping[tid], pos=(tower_data['masterPosX'], tower_data['masterPosY']), load=tower_data['masterLoad'],
                    type='tower', label=f'T{tid}')
        else:
            tower_data = timestamp_data[timestamp_data['towerId'] == tid][::-1].iloc[0]
            tower_features = [
                tower_data['towerPosX'], tower_data['towerPosY'],
                tower_data['towerLoad'], 0.0
            ]
            G.add_node(tower_mapping[tid], pos=(tower_data['towerPosX'], tower_data['towerPosY']), load=tower_data['towerLoad'],
                        type='tower', label=f'T{tid}')

        node_features.append(tower_features)

    # Add edges based on the masterId
    for _, row in timestamp_data.iterrows():
        vehicle_id = vehicle_mapping[row['vehicleId']]
        tower_id = tower_mapping[row['masterId']]
        edge_index.append([vehicle_id, tower_id])
        
        edge_features.append([row['throughput'], row['masterRssi'], row['masterDistance']])

        G.add_edge(vehicle_id, tower_id, throughput=row['throughput'], rssi=row['masterRssi'], distance=row['masterDistance'])
    
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)

    graph_data = Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_features_tensor)
    return graph_data, G, vehicle_mapping, tower_mapping

class GNNWithEdgeFeatures(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, out_channels):
        super(GNNWithEdgeFeatures, self).__init__()
        self.num_edge_features = num_edge_features
        self.conv1 = GCNConv(num_node_features + num_edge_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels + num_edge_features, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Aggregate edge features for each node
        row, col = edge_index
        agg_edge_features = torch.zeros(x.size(0), self.num_edge_features, device=x.device)
        agg_edge_features.index_add_(0, row, edge_attr)
        
        # Concatenate aggregated edge features with node features
        x = torch.cat([x, agg_edge_features], dim=-1)
        
        # First GCNConv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Concatenate edge features again
        x = torch.cat([x, agg_edge_features], dim=-1)
        
        # Second GCNConv layer
        x = self.conv2(x, edge_index)
        return x

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def get_triplets(data, embeddings, vehicle_mapping, tower_mapping):
    anchors = []
    positives = []
    negatives = []

    for vehicle_id in vehicle_mapping.values():
        connected_towers = data.edge_index[1][data.edge_index[0] == vehicle_id].tolist()

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

# Create graph data
graph_data, G, vehicle_mapping, tower_mapping = create_graph_data_for_timestamp(data, sim_timestamp)

# Create a DataLoader
loader = DataLoader([graph_data], batch_size=32, shuffle=True)

# Set up the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_node_features = graph_data.num_node_features
num_edge_features = graph_data.edge_attr.size(1)
model = GNNWithEdgeFeatures(num_node_features, num_edge_features, hidden_channels=64, out_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Check if a saved model exists, and load it
model_path = filepath + 'gnn_model.pth'
if os.path.exists(model_path):
    print("Loading the saved model for further training or inference...")
    model.load_state_dict(torch.load(model_path))
else:
    print("No saved model found. Training from scratch.")


# Training loop
for epoch in range(50):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index, data.edge_attr)
        
        anchors, positives, negatives = get_triplets(data, embeddings, vehicle_mapping, tower_mapping)
        loss = triplet_loss(anchors, positives, negatives)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')

# Save the model after training
torch.save(model.state_dict(), model_path)

# Extract embeddings
model.eval()
with torch.no_grad():
    embeddings = model(graph_data.x.to(device), graph_data.edge_index.to(device), graph_data.edge_attr.to(device))

vehicle_indices = list(vehicle_mapping.values())
tower_indices = list(tower_mapping.values())

vehicle_embeddings = embeddings[vehicle_indices]
tower_embeddings = embeddings[tower_indices]

vehicle_embeddings = vehicle_embeddings.cpu().numpy()
tower_embeddings = tower_embeddings.cpu().numpy()

# Calculate cosine similarity
similarity_matrix = cosine_similarity(vehicle_embeddings, tower_embeddings)

# Find top 3 candidates for each vehicle
top_3_candidates = {}
for i, vehicle_id in enumerate(vehicle_mapping.keys()):
    top_towers_indices = np.argsort(similarity_matrix[i])[::-1][:3]
    top_3_candidates[vehicle_id] = top_towers_indices

# Write results to file
with open(filepath +'outputGNN.txt', 'w+') as f:
    for vehicle_id, top_towers_indices in top_3_candidates.items():
        f.write(f'{vehicle_id}: {top_towers_indices}\n')

print("GNN processing completed. Results written to outputGNN.txt")