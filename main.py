from model import Encoder, corruption, Summarizer, cluster_net
from utils.load_data import load_data
from DGI import DeepGraphInfomax

import evaluation
import time

from sklearn.metrics import davies_bouldin_score
import networkx as nx
import numpy as np
import argparse
import torch
import os

# --- Arguments Configuration ---
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default="acm", help='cora, acm, citeseer, etc.')
parser.add_argument('--clustertemp', type=float, default=30, help='delta temperature for soft assignments')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')
args = parser.parse_args()

# --- Helper Functions ---
def make_modularity_matrix(adj):
    """Calculates the modularity matrix B = A - dd^T / 2M [cite: 317]"""
    adj = adj * (torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees @ degrees.t() / adj.sum()
    return mod

def result(graph, pred, labels):
    """Computes peak metrics: NMI, ACC, F1, ARI, and Hard Modularity Q [cite: 410, 413, 423]"""
    pred_np = pred.numpy()
    nmi = evaluation.NMI_helper(pred_np, labels)
    ac = evaluation.matched_ac(pred_np, labels)
    f1 = evaluation.cal_F_score(pred_np, labels)[0]
    ari = evaluation.adjusted_rand_score(pred_np, labels)
    q = evaluation.compute_modularity(graph, pred)
    return nmi, ac, f1, ari, q

def train(model, optimizer, feat, edge, selected_communities, adj, test_object):
    model.train()
    optimizer.zero_grad()
    pos_z, mu, r_mat, dist = model(feat, edge, selected_communities)
    # Using the paper's alpha = 0.001 scaling factor [cite: 323, 465, 764]
    modularity_loss = model.modularity(mu, r_mat, pos_z, dist, adj, test_object, args)
    loss = 0.001 * modularity_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, graph, feat, edge, selected_communities, label):
    model.eval()
    with torch.no_grad():
        node_emb, _, r_mat, _ = model(feat, edge, selected_communities)
    r_assign = r_mat.argmax(dim=1)
    r_nmi, r_ac, r_f1, r_ari, q = result(graph, r_assign, label)
    dbi = davies_bouldin_score(node_emb.cpu(), r_assign.cpu())
    return r_nmi, r_ac, r_f1, r_ari, dbi, q

# --- Main Sweep Execution ---
device = torch.device('cpu')
file_name = "sweep_results.csv"

# Load Data Once (Updated path to match your folder structure)
print(f"--- Loading {args.dataset} dataset from ./datasets/ ---")
data = load_data("./", args.dataset, "tensor", "npy", "npy", False, False, False, None)
feat = data.feature.type(torch.float32)
A = data.adj
adj = torch.tensor(A).type(torch.float32)
edge = torch.tensor(np.array(np.where(A == 1)))
test_object = make_modularity_matrix(adj)
label = data.label
graph = nx.from_numpy_array(A)

# Resolution Sweep range: 0.1 to 1.1 [cite: 756]
resolutions = np.arange(0.1, 1.2, 0.2)

for r_val in resolutions:
    print(f"\n[+] Testing Resolution: {r_val:.1f} -----------------------------")
    
    # 1. Generate Adaptive Scaffold [cite: 197, 212]
    structure_community = nx.community.louvain_communities(graph, resolution=r_val, seed=123)
    
    # 2. Size-based Filtering Strategy [cite: 218, 223]
    sizes = [len(c) for c in structure_community]
    threshold = np.mean(sizes) + 0.5 * np.std(sizes)
    selected_communities = [c for c in structure_community if len(c) > threshold]
    K = len(selected_communities)
    args.K = K
    print(f"   -> Selected K: {K} (from {len(structure_community)} raw clusters)")

    # 3. Model Initialization
    torch.manual_seed(args.seed)
    model = DeepGraphInfomax(
        hidden_channels=args.hidden, 
        encoder=Encoder(feat.shape[1], args.hidden),
        summary=Summarizer(), 
        corruption=corruption, 
        args=args, 
        cluster=cluster_net).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)

    m_nmi, m_ac, m_ari, m_f1, m_q = 0, 0, 0, 0, 0
    m_dbi = 3.0

    # 4. Training loop (300 iterations as per paper) [cite: 466]
    for epoch in range(1, 301):
        train(model, optimizer, feat, edge, selected_communities, adj, test_object)
        if epoch % 10 == 0:
            t_nmi, t_ac, t_f1, t_ari, t_dbi, t_q = test(model, graph, feat, edge, selected_communities, label)
            m_nmi, m_ac, m_f1, m_ari, m_q = max(m_nmi, t_nmi), max(m_ac, t_ac), max(m_f1, t_f1), max(m_ari, t_ari), max(m_q, t_q)
            m_dbi = min(m_dbi, t_dbi)

    # 5. Logging results
    with open(file_name, "a+") as f:
        log = (f"Dataset: {args.dataset} | Res: {r_val:.1f} | K: {K} | "
               f"NMI: {m_nmi:.4f} | ACC: {m_ac:.4f} | ARI: {m_ari:.4f} | Q: {m_q:.4f} | DBI: {m_dbi:.4f}\n")
        f.write(log)
        print(f"   -> {log.strip()}")

print(f"\nSweep complete. Results appended to {file_name}")