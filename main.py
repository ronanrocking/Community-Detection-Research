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

# For Leiden and Consensus logic
import igraph as ig
import leidenalg

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default="cora", help='Dataset name.')
parser.add_argument('--color', type=str, default='r-', help='color line')
parser.add_argument('--K', type=int, default=7, help='Number of partitions.')
parser.add_argument('--clustertemp', type=float, default=30, help='Softmax temperature.')
parser.add_argument('--train_iters', type=int, default=1001, help='Total training iterations.')
parser.add_argument('--num_cluster_iter', type=int, default=1, help='Clustering iterations.')
parser.add_argument('--seed', type=int, default=24, help='Base random seed.')
args = parser.parse_args()

# --- Core Helper Functions (Original Research Logic) ---
def make_modularity_matrix(adj):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod

def result(graph, pred, labels):
    pred_np = pred.numpy()
    nmi = evaluation.NMI_helper(pred_np, labels)
    ac = evaluation.matched_ac(pred_np, labels)
    f1 = evaluation.cal_F_score(pred_np, labels)[0]
    ari = evaluation.adjusted_rand_score(pred_np, labels)
    q = evaluation.compute_modularity(graph, pred)
    return nmi, ac, f1, ari, q

# --- NEW: CONSENSUS SCAFFOLD GENERATION ---
def get_consensus_scaffold(graph, algo_type, n_runs=15):
    """
    Runs clustering multiple times with varying seeds to find the 
    most stable community structure (Consensus).
    """
    num_nodes = graph.number_of_nodes()
    # Co-association matrix: tracks how many times nodes i and j share a community
    co_matrix = np.zeros((num_nodes, num_nodes))
    
    print(f"Generating Consensus {algo_type} Scaffold ({n_runs} runs)...")
    
    for i in range(n_runs):
        # We vary the seed each run to explore different local optima
        current_seed = args.seed + i 
        
        if algo_type == "Louvain":
            # Louvain is stochastic and sensitive to node processing order
            part = nx.community.louvain_communities(graph, resolution=0.3, seed=current_seed)
            membership = np.zeros(num_nodes)
            for cluster_id, nodes in enumerate(part):
                for node in nodes: membership[node] = cluster_id
        else:
            # Leiden also has a stochastic refinement phase
            g_ig = ig.Graph.from_networkx(graph)
            part = leidenalg.find_partition(g_ig, leidenalg.RBConfigurationVertexPartition, 
                                            resolution_parameter=0.3, seed=current_seed)
            membership = part.membership

        # Update the co-association matrix with pairs found in the same cluster
        for n1 in range(num_nodes):
            for n2 in range(n1 + 1, num_nodes):
                if membership[n1] == membership[n2]:
                    co_matrix[n1, n2] += 1
                    co_matrix[n2, n1] += 1

    # CONSENSUS THRESHOLD:
    # Only keep connections that appeared in > 50% of the runs (Majority Vote)
    consensus_adj = (co_matrix / n_runs) > 0.5
    consensus_graph = nx.from_numpy_array(consensus_adj.astype(int))
    
    # Final pass to extract the stable community structure
    consensus_communities = nx.community.louvain_communities(consensus_graph, resolution=0.3, seed=args.seed)
    return consensus_communities

# --- Main Execution Setup ---
#dataset_list = ["acm", "amac", "amap", "citeseer", "cocs", "cora", "film", "pubmed", "uat"]
dataset_list = ["cocs", "pubmed"]
device = torch.device('cpu')
b = 0.001 # Modularity loss weight from paper
file_name = "consensus_results.csv"

for ds in dataset_list:
    args.dataset = ds
    print(f"\n{'='*20} DATASET: {ds.upper()} {'='*20}")
    
    try:
        # Load Data
        data = load_data("./", ds, "tensor", "npy", "npy", False, False, False, None)
        feat, label = data.feature.type(torch.float32), data.label
        A = data.adj
        adj, edge = torch.tensor(A).type(torch.float32), torch.tensor(np.array(np.where(A == 1)))
        test_object, graph = make_modularity_matrix(adj), nx.from_numpy_array(A)

        for algo_name in ["Louvain","Leiden"]:
            start_total = time.perf_counter()
            
            # STEP 1: Build the Consensus Scaffold
            # This replaces the single-run "lucky" scaffold with a stable reference
            structure_community = get_consensus_scaffold(graph, algo_name, n_runs=15)

            # STEP 2: Paper's Structural Filtering Logic
            # Removes tiny outlier communities before calculating centers (mu)
            nums = [len(i) for i in structure_community]
            threshold = np.mean(nums) + 0.5 * np.std(nums)
            selected_communities = [c for c in structure_community if len(c) > threshold]
            K = len(selected_communities)
            args.K = K

            # STEP 3: Model Training (Exactly as original)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            model = DeepGraphInfomax(hidden_channels=args.hidden, encoder=Encoder(feat.shape[1], args.hidden), 
                                     summary=Summarizer(), corruption=corruption, args=args, cluster=cluster_net).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)

            max_nmi, max_ac, max_ari, max_f1, max_q, min_dbi = 0, 0, 0, 0, 0, 3
            patience, stop_cnt, min_loss = 200, 0, 1e9

            print(f"Training with {K} stable communities...")
            for epoch in range(1, 301):
                model.train()
                optimizer.zero_grad()
                pos_z, mu, r, dist = model(feat, edge, selected_communities)
                loss = b * model.modularity(mu, r, pos_z, dist, adj, test_object, args)
                loss.backward()
                optimizer.step()

                # Test and keep best metrics
                if epoch % 2 == 0:
                    model.eval()
                    with torch.no_grad():
                        node_emb, _, r_val, _ = model(feat, edge, selected_communities)
                    r_assign = r_val.argmax(dim=1)
                    t_nmi, t_ac, t_f1, t_ari, t_q = result(graph, r_assign, label)
                    t_dbi = davies_bouldin_score(node_emb, r_assign)
                    
                    max_nmi, max_ac, max_f1, max_ari, max_q = max(max_nmi, t_nmi), max(max_ac, t_ac), max(max_f1, t_f1), max(max_ari, t_ari), max(max_q, t_q)
                    min_dbi = min(min_dbi, t_dbi)

                if loss < min_loss: min_loss, stop_cnt = loss, 0
                else: stop_cnt += 1
                if stop_cnt >= patience: break

            # Save Results
            end_total = time.perf_counter()
            with open(file_name, "a+") as f:
                f.write(f"{ds},{algo_name},{K},{max_nmi:.4f},{max_ac:.4f},{max_f1:.4f},{max_ari:.4f},{min_dbi:.4f},{max_q:.4f},{end_total-start_total:.2f}\n")
            
            print(f"Finished {algo_name}. Max NMI: {max_nmi:.4f}")

    except Exception as e:
        print(f"Error on {ds}: {e}")