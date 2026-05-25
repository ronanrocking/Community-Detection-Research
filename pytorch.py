import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, davies_bouldin_score
import numpy as np

# --- 1. DATA SETUP ---
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# --- 2. MODULARITY FUNCTION (Paper Eq. 3) ---
def calculate_modularity(preds, edge_index, num_nodes):
    """Calculates Q based on the paper's definition """
    A = to_dense_adj(edge_index)[0].numpy()
    M = edge_index.shape[1] / 2  # Total edges [cite: 165]
    degrees = A.sum(axis=1)
    
    Q = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if preds[i] == preds[j]: # Kronecker delta [cite: 194]
                # (a_ij - (di*dj)/2M) 
                Q += (A[i,j] - (degrees[i] * degrees[j]) / (2 * M))
    return Q / (2 * M)

# --- 3. MODEL DEFINITION ---
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # We need the first layer output for DBI 
        h = F.relu(self.conv1(x, edge_index))
        x = F.dropout(h, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), h

# --- 4. TRAINING ---
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(201):
    optimizer.zero_grad()
    out, _ = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f}')

# --- 5. FULL METRIC EVALUATION ---
model.eval()
with torch.no_grad():
    logits, embeddings = model(data)
    preds = logits.argmax(dim=1).numpy()
    labels = data.y.numpy()
    mask = data.test_mask.numpy()

    # Similarity to Ground Truth (Higher is Better)
    acc = (preds[mask] == labels[mask]).sum() / mask.sum() [cite: 432]
    nmi = normalized_mutual_info_score(labels[mask], preds[mask]) [cite: 416]
    ari = adjusted_rand_score(labels[mask], preds[mask]) [cite: 443]
    f1 = f1_score(labels[mask], preds[mask], average='macro') [cite: 437]

    # Structural Metrics
    # DBI uses the hidden embeddings  (Lower is Better )
    dbi = davies_bouldin_score(embeddings.numpy()[mask], preds[mask])
    
    # Q (Modularity) measures cluster density  (Higher is Better)
    q_score = calculate_modularity(preds, data.edge_index, data.num_nodes) [cite: 190]

print("\n--- FINAL RESEARCH METRICS ---")
print(f"Accuracy:   {acc:.4f}")
print(f"NMI:        {nmi:.4f}")
print(f"ARI:        {ari:.4f}")
print(f"F1-Macro:   {f1:.4f}")
print(f"DBI:        {dbi:.4f} (Lower is better)")
print(f"Modularity: {q_score:.4f} (Q)")