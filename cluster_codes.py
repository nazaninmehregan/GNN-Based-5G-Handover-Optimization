import torch
import torch_cluster
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_cluster import random_walk
from sklearn.cluster import KMeans

# print(embeddings.size(1))
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_assignments = kmeans.fit_predict(embeddings.cpu().numpy())
cluster_assignments = torch.tensor(cluster_assignments, dtype=torch.long)

import numpy as np
from sklearn.cluster import KMeans

# model.eval()
# embeddings = []
# for data in loader:
#     data = data.to(device)
#     emb = model(data.x, data.edge_index).detach().cpu().numpy()
#     embeddings.append(emb)

embeddings = np.concatenate(embeddings, axis=0)

kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Scale embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Visualize using t-SNE
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(embeddings_scaled)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.show()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=7, random_state=0).fit(embeddings_scaled)
labels = kmeans.labels_

# Visualize the clusters
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)
plt.show()