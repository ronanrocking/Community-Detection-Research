import argparse
from itertools import combinations

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np

import evaluation
from utils.load_data import load_data


def run_community_detection(graph, algo_type, seed):
    num_nodes = graph.number_of_nodes()

    if algo_type == "Louvain":
        part = nx.community.louvain_communities(graph, resolution=0.3, seed=seed)
        membership = np.zeros(num_nodes, dtype=int)
        for cluster_id, nodes in enumerate(part):
            for node in nodes:
                membership[node] = cluster_id
        communities = [set(nodes) for nodes in part]
    else:
        g_ig = ig.Graph.from_networkx(graph)
        part = leidenalg.find_partition(
            g_ig,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=0.3,
            seed=seed,
        )
        membership = np.array(part.membership, dtype=int)
        communities_by_id = {}
        for node, cluster_id in enumerate(membership):
            communities_by_id.setdefault(cluster_id, set()).add(node)
        communities = list(communities_by_id.values())

    return membership, communities


def get_usable_k(communities):
    if not communities:
        return 0

    nums = [len(i) for i in communities]
    threshold = np.mean(nums) + 0.5 * np.std(nums)
    return len([c for c in communities if len(c) > threshold])


def get_community_balance(communities, num_nodes):
    if not communities or num_nodes == 0:
        return 0.0

    sizes = [len(c) for c in communities]
    largest_ratio = max(sizes) / num_nodes
    tiny_threshold = max(3, 0.005 * num_nodes)
    tiny_ratio = len([size for size in sizes if size < tiny_threshold]) / len(sizes)
    balance = 1 - 0.5 * largest_ratio - 0.5 * tiny_ratio
    return float(np.clip(balance, 0.0, 1.0))


def score_algorithm_candidate(graph, algo_type, seed, probe_runs):
    memberships = []
    usable_ks = []
    balances = []

    for i in range(probe_runs):
        membership, communities = run_community_detection(graph, algo_type, seed + i)
        memberships.append(membership)
        usable_ks.append(get_usable_k(communities))
        balances.append(get_community_balance(communities, graph.number_of_nodes()))

    pair_scores = [
        evaluation.NMI_helper(first, second)
        for first, second in combinations(memberships, 2)
    ]
    stability = float(np.mean(pair_scores)) if pair_scores else 0.0
    k_stability = float(1 / (1 + np.std(usable_ks)))
    balance = float(np.mean(balances)) if balances else 0.0
    score = 0.60 * stability + 0.25 * k_stability + 0.15 * balance

    return {
        "algorithm": algo_type,
        "score": score,
        "stability": stability,
        "k_stability": k_stability,
        "balance": balance,
        "usable_ks": usable_ks,
    }


def choose_scaffold_algorithm(graph, seed, probe_runs):
    candidates = [
        score_algorithm_candidate(graph, "Louvain", seed, probe_runs),
        score_algorithm_candidate(graph, "Leiden", seed, probe_runs),
    ]
    return max(candidates, key=lambda item: item["score"]), candidates


def main():
    parser = argparse.ArgumentParser(
        description="Probe the adaptive Louvain/Leiden selector without running consensus training."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["acm", "citeseer", "cora", "amap", "amac"],
        help="Datasets to test.",
    )
    parser.add_argument("--seed", type=int, default=24, help="Base random seed.")
    parser.add_argument("--probe-runs", type=int, default=3, help="Probe runs per algorithm.")
    args = parser.parse_args()

    print("Adaptive selector dry run only. No 15-run consensus or GNN training is executed.")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Seed: {args.seed}, probe_runs: {args.probe_runs}\n")

    for dataset in args.datasets:
        print(f"{'=' * 20} DATASET: {dataset.upper()} {'=' * 20}")
        data = load_data("./", dataset, "tensor", "npy", "npy", False, False, False, None)
        graph = nx.from_numpy_array(data.adj)
        best, candidates = choose_scaffold_algorithm(graph, args.seed, args.probe_runs)

        for candidate in candidates:
            print(
                f"  {candidate['algorithm']}: score={candidate['score']:.4f}, "
                f"stability={candidate['stability']:.4f}, "
                f"K-stability={candidate['k_stability']:.4f}, "
                f"balance={candidate['balance']:.4f}, "
                f"probe_Ks={candidate['usable_ks']}"
            )
        print(f"Chosen scaffold algorithm: {best['algorithm']}\n")


if __name__ == "__main__":
    main()
