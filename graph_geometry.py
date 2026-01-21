import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import ot
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm
import skdim
import os

# --- STEP 1: LOAD DATASET ---
print("Step 1: Loading MUTAG dataset...")
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
graphs = [to_networkx(d, node_attrs=['x']) for d in dataset[:100]] # Using 100 samples
labels = [dataset[i].y.item() for i in range(100)]

# --- STEP 2: DISTANCE CALCULATION FUNCTION ---
def compute_fgw(g1, g2):
    adj1, adj2 = nx.to_numpy_array(g1), nx.to_numpy_array(g2)
    feat1 = np.array([d for _, d in g1.nodes(data='x')])
    feat2 = np.array([d for _, d in g2.nodes(data='x')])
    
    C1, C2 = adj1, adj2
    M = ot.dist(feat1, feat2, metric='euclidean')
    p1, p2 = ot.unif(len(g1)), ot.unif(len(g2))
    
    # alpha=0.5 balances structure and features
    val = ot.gromov.fused_gromov_wasserstein2(M, C1, C2, p1, p2, alpha=0.5)
    return np.sqrt(max(0, val))

# --- STEP 3: COMPUTE MATRIX ---
n = len(graphs)
dist_matrix = np.zeros((n, n))
print(f"Step 2: Computing {n}x{n} FGW Distance Matrix...")
for i in tqdm(range(n)):
    for j in range(i + 1, n):
        d = compute_fgw(graphs[i], graphs[j])
        dist_matrix[i, j] = dist_matrix[j, i] = d

# --- STEP 4: MDS VISUALIZATION ---
print("Step 3: Generating MDS Plot...")
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(dist_matrix)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', edgecolors='k', alpha=0.7)
plt.title("MDS Projection (Manifold Fragmentation)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

# --- STEP 5: INTRINSIC DIMENSION (MLE) ---
print("Step 4: Estimating Intrinsic Dimension...")
# We use the distances to estimate local dimensionality
mle = skdim.id.MLE()
local_ids = mle.fit_transform(dist_matrix)
mean_id = np.mean(local_ids)

plt.subplot(1, 2, 2)
plt.hist(local_ids, bins=15, color='teal', edgecolor='black', alpha=0.7)
plt.axvline(mean_id, color='red', linestyle='--', label=f'Mean ID: {mean_id:.2f}')
plt.title("Intrinsic Dimension Distribution")
plt.xlabel("Dimension")
plt.legend()

plt.tight_layout()
print(f"\nAnalysis Complete. Mean Intrinsic Dimension: {mean_id:.2f}")
plt.show()