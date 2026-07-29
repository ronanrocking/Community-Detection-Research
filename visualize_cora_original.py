"""Create a Cora t-SNE using the training logic in Original/main.py."""

from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE

from utils.load_data import load_data
from visualize_citeseer_original import (
    select_original_communities,
    train_original_embeddings,
)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    best_epoch: int,
    output: Path,
) -> None:
    coordinates = TSNE(
        n_components=2,
        perplexity=30.0,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=24,
    ).fit_transform(embeddings)

    output.parent.mkdir(parents=True, exist_ok=True)
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
        title="Cora class",
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )
    ax.set_title(
        "Cora node embeddings — t-SNE\n"
        f"Original Louvain scaffold, K={k}, best epoch={best_epoch}"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        output.with_suffix(".npz"),
        coordinates=coordinates,
        labels=labels,
        best_epoch=best_epoch,
    )


def main() -> None:
    args = Namespace(lr=0.001, hidden=512, clustertemp=30.0, seed=24)
    data = load_data(
        "./", "cora", "tensor", "npy", "npy", False, False, False
    )
    features = data.feature.to(dtype=torch.float32)
    labels = np.asarray(data.label)
    adjacency_np = np.asarray(data.adj)
    adjacency = torch.as_tensor(adjacency_np, dtype=torch.float32)
    edge_index = torch.as_tensor(np.array(np.where(adjacency_np == 1)))
    graph = nx.from_numpy_array(adjacency_np)

    communities = select_original_communities(graph)
    print(
        f"Original Louvain selected {len(communities)} communities: "
        f"{[len(community) for community in communities]}"
    )
    embeddings, best_epoch = train_original_embeddings(
        features, edge_index, adjacency, communities, args
    )
    output = Path("visualizations") / "cora_original_tsne.png"
    plot_tsne(embeddings, labels, len(communities), best_epoch, output)
    print(f"Best epoch: {best_epoch}")
    print(f"Saved plot: {output.resolve()}")
    print(f"Saved coordinates: {output.with_suffix('.npz').resolve()}")


if __name__ == "__main__":
    main()
