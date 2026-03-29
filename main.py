from model import Encoder, corruption, Summarizer, cluster_net
from utils.load_data import load_data
from DGI import DeepGraphInfomax

import evaluation
import time
import torch.nn.functional as F
from sklearn.metrics import davies_bouldin_score
import networkx as nx
import numpy as np
import argparse
import torch
import os

# For Leiden and Consensus logic
import igraph as ig
import leidenalg

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default="cora", help='Dataset name.')
parser.add_argument('--clustertemp', type=float, default=30, help='Softmax temperature.')
parser.add_argument('--train_iters', type=int, default=1001, help='Total training iterations.')
parser.add_argument('--seed', type=int, default=24, help='Base random seed.')
args = parser.parse_args()

# --- Core Helper Functions ---
def make_modularity_matrix(adj):
    adj = adj * (torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees @ degrees.t() / adj.sum()
    return mod

def result(graph, pred, labels):
    pred_np = pred.numpy()
    nmi = evaluation.NMI_helper(pred_np, labels)
    ac = evaluation.matched_ac(pred_np, labels)
    f1 = evaluation.cal_F_score(pred_np, labels)[0]
    ari = evaluation.adjusted_rand_score(pred_np, labels)
    q = evaluation.compute_modularity(graph, pred)
    return nmi, ac, f1, ari, q

# --- CONSENSUS SCAFFOLD (n=1) ---
def get_consensus_scaffold(graph, algo_type, n_runs=1):
    num_nodes = graph.number_of_nodes()
    co_matrix = np.zeros((num_nodes, num_nodes))
    
    print(f"Generating Consensus {algo_type} Scaffold | Runs: {n_runs}")
    
    for i in range(n_runs):
        current_seed = args.seed + i 
        if algo_type == "Louvain":
            part = nx.community.louvain_communities(graph, resolution=0.3, seed=current_seed)
            membership = np.zeros(num_nodes)
            for cluster_id, nodes in enumerate(part):
                for node in nodes: membership[node] = cluster_id
        else:
            g_ig = ig.Graph.from_networkx(graph)
            part = leidenalg.find_partition(g_ig, leidenalg.RBConfigurationVertexPartition, 
                                            resolution_parameter=0.3, seed=current_seed)
            membership = part.membership

        for n1 in range(num_nodes):
            for n2 in graph.neighbors(n1):
                if n2 > n1 and membership[n1] == membership[n2]:
                    co_matrix[n1, n2] += 1
                    co_matrix[n2, n1] += 1

    consensus_adj = np.zeros((num_nodes, num_nodes))
    rows, cols = np.where(co_matrix > (n_runs / 2))
    for r, c in zip(rows, cols):
        if r < c:
            consensus_adj[r, c] = 1
            consensus_adj[c, r] = 1

    consensus_graph = nx.from_numpy_array(consensus_adj)
    
    if algo_type == "Louvain":
        return nx.community.louvain_communities(consensus_graph, resolution=0.3, seed=args.seed)
    else:
        g_ig_final = ig.Graph.from_networkx(consensus_graph)
        part_final = leidenalg.find_partition(g_ig_final, leidenalg.RBConfigurationVertexPartition, 
                                              resolution_parameter=0.3, seed=args.seed)
        final_comm = [set() for _ in range(len(set(part_final.membership)))]
        for node_idx, cluster_idx in enumerate(part_final.membership):
            final_comm[cluster_idx].add(node_idx)
        return final_comm

# --- SOFT DENSITY SELECTION ---
def select_elite_communities_soft(graph, communities):
    selected = []
    stats = [] 
    
    for comm in communities:
        if len(comm) < 5: continue 
        
        subgraph = graph.subgraph(comm)
        actual_edges = subgraph.number_of_edges()
        possible_edges = (len(comm) * (len(comm) - 1)) / 2
        
        density = actual_edges / possible_edges if possible_edges > 0 else 0
        stats.append((list(comm), density))
    
    if not stats: return []

    all_densities = [s[1] for s in stats]
    mean_d = np.mean(all_densities)
    std_d = np.std(all_densities)
    
    # NEW THRESHOLD: Mean minus half a standard deviation
    threshold = mean_d - (0.5 * std_d)
    
    print(f"--- Density Stats | Mean: {mean_d:.4f} | SD: {std_d:.4f} | Threshold: {threshold:.4f} ---")
    
    for nodes, d in stats:
        if d >= threshold:
            selected.append(nodes)
            
    return selected

# --- Main Execution ---
dataset_list = ["acm", "amac", "amap", "citeseer", "cora", "film", "pubmed"]
device = torch.device('cpu')
b = 0.001 
file_name = "density_soft_results.csv"


for ds in dataset_list:
    args.dataset = ds
    print(f"\n{'='*15} DATASET: {ds.upper()} {'='*15}")
    
    try:
        data = load_data("./", ds, "tensor", "npy", "npy", False, False, False, None)
        feat, label = data.feature.type(torch.float32), data.label
        A = data.adj
        adj, edge = torch.tensor(A).type(torch.float32), torch.tensor(np.array(np.where(A == 1)))
        test_object, graph = make_modularity_matrix(adj), nx.from_numpy_array(A)

        for algo_name in ["Louvain", "Leiden"]:
            start_total = time.perf_counter()
            
            # 1. Clustering
            raw_communities = get_consensus_scaffold(graph, algo_name, n_runs=1)

            # 2. Soft Thresholding
            selected_communities = select_elite_communities_soft(graph, raw_communities)
            
            K = len(selected_communities)
            args.K = K
            
            if K < 2:
                print(f"Skipping {algo_name} on {ds}: K={K} (Insufficient clusters).")
                continue

            # 3. Model Training
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            model = DeepGraphInfomax(hidden_channels=512, encoder=Encoder(feat.shape[1], 512), 
                                     summary=Summarizer(), corruption=corruption, args=args, cluster=cluster_net).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)

            max_nmi, max_ac, max_f1, max_ari, max_q, min_dbi = 0, 0, 0, 0, 0, 3
            patience, stop_cnt, min_loss = 200, 0, 1e9

            print(f"Training {algo_name} | Soft K: {K}")
            for epoch in range(1, 301):
                model.train()
                optimizer.zero_grad()
                pos_z, mu, r, dist = model(feat, edge, selected_communities)
                loss = b * model.modularity(mu, r, pos_z, dist, adj, test_object, args)
                loss.backward()
                optimizer.step()

                if epoch % 2 == 0:
                    model.eval()
                    with torch.no_grad():
                        node_emb, _, r_val, _ = model(feat, edge, selected_communities)
                    
                    r_assign = r_val.argmax(dim=1)
                    if len(torch.unique(r_assign)) < 2: continue

                    t_nmi, t_ac, t_f1, t_ari, t_q = result(graph, r_assign, label)
                    t_dbi = davies_bouldin_score(node_emb, r_assign)
                    
                    max_nmi, max_ac, max_f1, max_ari, max_q = max(max_nmi, t_nmi), max(max_ac, t_ac), max(max_f1, t_f1), max(max_ari, t_ari), max(max_q, t_q)
                    min_dbi = min(min_dbi, t_dbi)

                if loss < min_loss: min_loss, stop_cnt = loss, 0
                else: stop_cnt += 1
                if stop_cnt >= patience: break

            end_total = time.perf_counter()
            with open(file_name, "a+") as f:
                f.write(f"{ds},{algo_name},{K},{max_nmi:.4f},{max_ac:.4f},{max_f1:.4f},{max_ari:.4f},{min_dbi:.4f},{max_q:.4f},{end_total-start_total:.2f}\n")
            
            print(f"Done {algo_name}. Soft K: {K} | Max NMI: {max_nmi:.4f}")

    except Exception as e:
        print(f"Error on {ds}: {e}")