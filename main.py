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

# Added for Leiden
import igraph as ig
import leidenalg

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default="cora", help='default dataset (overridden by loop)')
parser.add_argument('--color', type=str, default='r-', help='color line')
parser.add_argument('--K', type=int, default=7, help='How many partitions')
parser.add_argument('--clustertemp', type=float, default=30, help='how hard to make the softmax')
parser.add_argument('--train_iters', type=int, default=1001, help='number of training iterations')
parser.add_argument('--num_cluster_iter', type=int, default=1, help='number of iterations for clustering')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')
args = parser.parse_args()

# --- Original Functions (Untouched) ---
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

# --- Dataset List from your folder structure ---
dataset_list = ["acm", "amac", "amap", "citeseer", "cocs", "cora", "film", "pubmed", "uat"]

device = torch.device('cpu')
b = 0.001
file_name = "result.csv"

# --- Main Execution Loop ---
for ds in dataset_list:
    args.dataset = ds
    print(f"\n{'='*20} PROCESSING DATASET: {ds.upper()} {'='*20}")
    
    try:
        # Load Data
        data = load_data("./", args.dataset, "tensor", "npy", "npy", False, False, False, None)
        feat = data.feature.type(torch.float32)
        A = data.adj
        adj = torch.tensor(A).type(torch.float32)
        edge = torch.tensor(np.array(np.where(A == 1)))
        test_object = make_modularity_matrix(adj)
        label = data.label
        graph = nx.from_numpy_array(A)

        for algo_name in ["Louvain", "Leiden"]:
            print(f"Running {algo_name} on {ds}...")
            start_time = time.perf_counter()
            
            if algo_name == "Louvain":
                structure_community = nx.community.louvain_communities(graph, resolution=0.3, seed=123)
            else:
                g_ig = ig.Graph.from_networkx(graph)
                partition = leidenalg.find_partition(g_ig, leidenalg.RBConfigurationVertexPartition, resolution_parameter=0.3, seed=args.seed)
                membership = partition.membership
                structure_community = [{i for i, m in enumerate(membership) if m == cid} for cid in range(len(partition))]

            # Filtering Logic
            nums = [len(i) for i in structure_community]
            threshold = np.mean(nums) + 0.5 * np.std(nums)
            selected_communities = [c for c in structure_community if len(c) > threshold]
            K = len(selected_communities)
            args.K = K

            # Model Init
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            model = DeepGraphInfomax(hidden_channels=args.hidden, encoder=Encoder(feat.shape[1], args.hidden), 
                                     summary=Summarizer(), corruption=corruption, args=args, cluster=cluster_net).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)

            max_nmi, max_ac, max_ari, max_f1, max_q, min_dbi = 0, 0, 0, 0, 0, 3
            stop_cnt, patience, min_loss = 0, 200, 1e9

            # Training
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
                    t_nmi, t_ac, t_f1, t_ari, t_q = result(graph, r_assign, label)
                    t_dbi = davies_bouldin_score(node_emb, r_assign)
                    
                    max_nmi, max_ac, max_f1, max_ari, max_q = max(max_nmi, t_nmi), max(max_ac, t_ac), max(max_f1, t_f1), max(max_ari, t_ari), max(max_q, t_q)
                    min_dbi = min(min_dbi, t_dbi)

                if loss < min_loss:
                    min_loss, stop_cnt = loss, 0
                else:
                    stop_cnt += 1
                if stop_cnt >= patience: break

            # Save results
            end_time = time.perf_counter()
            with open(file_name, "a+") as f:
                f.write(f"{ds},{algo_name},{K},{max_nmi:.4f},{max_ac:.4f},{max_f1:.4f},{max_ari:.4f},{min_dbi:.4f},{max_q:.4f},{end_time-start_time:.2f}\n")
            
            print(f"  > {algo_name} Finished. NMI: {max_nmi:.4f}")

    except Exception as e:
        print(f"Error processing {ds}: {e}")
        continue