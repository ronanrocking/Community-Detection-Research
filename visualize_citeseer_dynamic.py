"""Create a Citeseer t-SNE using the training logic in 3#dynamicAlgo/main.py."""

from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE

from utils.load_data import load_data
from visualize_cora_tsne import (
    choose_algorithm,
    consensus_communities,
    select_large_communities,
    train_embeddings,
)


def plot_dynamic_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    k: int,
    args: Namespace,
) -> None:
    coordinates = TSNE(
        n_components=2,
        perplexity=args.perplexity,
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
    ax.legend(
        *scatter.legend_elements(num=len(np.unique(labels))),
        title="Citeseer class",
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )
    ax.set_title(
        "Citeseer node embeddings — t-SNE\n"
        f"{algorithm} consensus scaffold, K={k}, seed={args.seed}"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.15)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        coordinates=coordinates,
        labels=labels,
    )
    print(f"Saved plot: {args.output.resolve()}")
    print(f"Saved coordinates: {args.output.with_suffix('.npz').resolve()}")


def main() -> None:
    args = Namespace(
        lr=0.001,
        hidden=512,
        clustertemp=30.0,
        seed=24,
        epochs=300,
        patience=200,
        probe_runs=3,
        consensus_runs=15,
        perplexity=30.0,
        output=Path("visualizations") / "citeseer_dynamic_tsne.png",
    )

    data = load_data(
        "./", "citeseer", "tensor", "npy", "npy", False, False, False
    )
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
    plot_dynamic_tsne(
        embeddings, labels, algorithm, len(selected), args
    )


if __name__ == "__main__":
    main()
