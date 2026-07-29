"""Train the Cora model used by main.py and plot its embeddings with t-SNE.

This is intentionally standalone: importing main.py would execute its dataset
training loop.  The defaults mirror main.py's Cora-relevant settings.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE

import evaluation
from DGI import DeepGraphInfomax
from model import Encoder, Summarizer, cluster_net, corruption
from utils.load_data import load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the main.py model on Cora and create a t-SNE plot."
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--clustertemp", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--probe-runs", type=int, default=3)
    parser.add_argument("--consensus-runs", type=int, default=15)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualizations") / "cora_tsne.png",
    )
    return parser.parse_args()


def make_modularity_matrix(adj: torch.Tensor) -> torch.Tensor:
    no_self_loops = adj * (torch.ones_like(adj) - torch.eye(adj.shape[0]))
    degrees = no_self_loops.sum(dim=0).unsqueeze(1)
    return no_self_loops - degrees @ degrees.t() / no_self_loops.sum()


def run_community_detection(
    graph: nx.Graph, algorithm: str, seed: int
) -> tuple[np.ndarray, list[set[int]]]:
    if algorithm == "Louvain":
        communities = [
            set(nodes)
            for nodes in nx.community.louvain_communities(
                graph, resolution=0.3, seed=seed
            )
        ]
        membership = np.zeros(graph.number_of_nodes(), dtype=int)
        for cluster_id, nodes in enumerate(communities):
            membership[list(nodes)] = cluster_id
        return membership, communities

    igraph_graph = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=0.3,
        seed=seed,
    )
    membership = np.asarray(partition.membership, dtype=int)
    communities_by_id: dict[int, set[int]] = {}
    for node, cluster_id in enumerate(membership):
        communities_by_id.setdefault(cluster_id, set()).add(node)
    return membership, list(communities_by_id.values())


def usable_k(communities: list[set[int]]) -> int:
    sizes = np.asarray([len(community) for community in communities])
    threshold = sizes.mean() + 0.5 * sizes.std()
    return int((sizes > threshold).sum())


def community_balance(communities: list[set[int]], num_nodes: int) -> float:
    sizes = np.asarray([len(community) for community in communities])
    largest_ratio = sizes.max() / num_nodes
    tiny_threshold = max(3, 0.005 * num_nodes)
    tiny_ratio = (sizes < tiny_threshold).sum() / len(sizes)
    return float(np.clip(1 - 0.5 * largest_ratio - 0.5 * tiny_ratio, 0, 1))


def score_algorithm(
    graph: nx.Graph, algorithm: str, seed: int, probe_runs: int
) -> dict[str, object]:
    memberships: list[np.ndarray] = []
    usable_ks: list[int] = []
    balances: list[float] = []
    for offset in range(probe_runs):
        membership, communities = run_community_detection(
            graph, algorithm, seed + offset
        )
        memberships.append(membership)
        usable_ks.append(usable_k(communities))
        balances.append(community_balance(communities, graph.number_of_nodes()))

    pair_scores = [
        evaluation.NMI_helper(first, second)
        for first, second in combinations(memberships, 2)
    ]
    stability = float(np.mean(pair_scores)) if pair_scores else 0.0
    k_stability = float(1 / (1 + np.std(usable_ks)))
    balance = float(np.mean(balances))
    score = 0.60 * stability + 0.25 * k_stability + 0.15 * balance
    return {
        "algorithm": algorithm,
        "score": score,
        "stability": stability,
        "k_stability": k_stability,
        "balance": balance,
        "usable_ks": usable_ks,
    }


def choose_algorithm(
    graph: nx.Graph, seed: int, probe_runs: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates = [
        score_algorithm(graph, algorithm, seed, probe_runs)
        for algorithm in ("Louvain", "Leiden")
    ]
    return max(candidates, key=lambda candidate: candidate["score"]), candidates


def consensus_communities(
    graph: nx.Graph, algorithm: str, seed: int, runs: int
) -> list[set[int]]:
    memberships = [
        run_community_detection(graph, algorithm, seed + offset)[0]
        for offset in range(runs)
    ]
    num_nodes = graph.number_of_nodes()
    co_matrix = np.zeros((num_nodes, num_nodes), dtype=np.uint16)
    for membership in memberships:
        for cluster_id in np.unique(membership):
            nodes = np.flatnonzero(membership == cluster_id)
            co_matrix[np.ix_(nodes, nodes)] += 1

    consensus_adj = (co_matrix / runs) > 0.5
    np.fill_diagonal(consensus_adj, False)
    consensus_graph = nx.from_numpy_array(consensus_adj)
    return [
        set(nodes)
        for nodes in nx.community.louvain_communities(
            consensus_graph, resolution=0.3, seed=seed
        )
    ]


def select_large_communities(
    communities: list[set[int]],
) -> list[set[int]]:
    sizes = np.asarray([len(community) for community in communities])
    threshold = sizes.mean() + 0.5 * sizes.std()
    selected = [
        community
        for community, size in zip(communities, sizes)
        if size > threshold
    ]
    if not selected:
        raise RuntimeError("Structural filtering produced no usable communities.")
    return selected


def train_embeddings(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    adjacency: torch.Tensor,
    communities: list[set[int]],
    args: argparse.Namespace,
) -> np.ndarray:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # DeepGraphInfomax reads these fields from the same argparse-style object
    # used by main.py.
    args.K = len(communities)
    args.num_cluster_iter = 1
    model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=Encoder(features.shape[1], args.hidden, p_feat=0.25),
        summary=Summarizer(),
        corruption=corruption,
        args=args,
        cluster=cluster_net,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
    modularity_matrix = make_modularity_matrix(adjacency)
    minimum_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        embeddings, centers, assignments, distances = model(
            features, edge_index, communities
        )
        loss = 0.001 * model.modularity(
            centers,
            assignments,
            embeddings,
            distances,
            adjacency,
            modularity_matrix,
            args,
        )
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach())
        if loss_value < minimum_loss:
            minimum_loss = loss_value
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:3d}/{args.epochs}: loss={loss_value:.8f}")
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.eval()
    with torch.no_grad():
        embeddings, _, _, _ = model(features, edge_index, communities)
    return embeddings.cpu().numpy()


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    k: int,
    args: argparse.Namespace,
) -> None:
    perplexity = min(args.perplexity, embeddings.shape[0] - 1)
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=args.seed,
    ).fit_transform(embeddings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab10",
        s=14,
        alpha=0.82,
        linewidths=0,
    )
    legend = ax.legend(
        *scatter.legend_elements(num=len(np.unique(labels))),
        title="Cora class",
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )
    ax.add_artist(legend)
    ax.set_title(
        f"Cora node embeddings — t-SNE\n"
        f"{algorithm} consensus scaffold, K={k}, seed={args.seed}"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.15)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    coordinates_path = args.output.with_suffix(".npz")
    np.savez_compressed(coordinates_path, coordinates=coordinates, labels=labels)
    print(f"Saved plot: {args.output.resolve()}")
    print(f"Saved coordinates: {coordinates_path.resolve()}")


def main() -> None:
    args = parse_args()
    data = load_data("./", "cora", "tensor", "npy", "npy", False, False, False)
    features = data.feature.to(dtype=torch.float32)
    labels = np.asarray(data.label)
    adjacency_np = np.asarray(data.adj)
    adjacency = torch.as_tensor(adjacency_np, dtype=torch.float32)
    edge_index = torch.as_tensor(np.array(np.where(adjacency_np == 1)))
    graph = nx.from_numpy_array(adjacency_np)

    best, candidates = choose_algorithm(graph, args.seed, args.probe_runs)
    for candidate in candidates:
        print(
            f"{candidate['algorithm']}: score={candidate['score']:.4f}, "
            f"stability={candidate['stability']:.4f}, "
            f"K-stability={candidate['k_stability']:.4f}, "
            f"balance={candidate['balance']:.4f}, "
            f"probe_Ks={candidate['usable_ks']}"
        )
    algorithm = str(best["algorithm"])
    print(f"Chosen scaffold algorithm: {algorithm}")

    consensus = consensus_communities(
        graph, algorithm, args.seed, args.consensus_runs
    )
    selected = select_large_communities(consensus)
    print(f"Training with {len(selected)} stable communities (p_feat=0.25).")
    embeddings = train_embeddings(
        features, edge_index, adjacency, selected, args
    )
    plot_tsne(embeddings, labels, algorithm, len(selected), args)


if __name__ == "__main__":
    main()
