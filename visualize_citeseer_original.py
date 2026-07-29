"""Create a Citeseer t-SNE using the training logic in Original/main.py."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset, uniform

from utils.load_data import load_data


class OriginalEncoder(nn.Module):
    """Encoder from model.py on the Original branch."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        structure_center: list[set[int]],
    ) -> torch.Tensor:
        del structure_center
        return self.prelu(self.conv(x, edge_index))


class OriginalSummarizer(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z.mean(dim=0))


def original_corruption(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    structure_center: list[set[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    del structure_center
    return x[torch.randperm(x.size(0))], edge_index


class OriginalDeepGraphInfomax(nn.Module):
    """Required Original/DGI.py behavior, including its normalization."""

    def __init__(
        self,
        hidden_channels: int,
        encoder: OriginalEncoder,
        summary: OriginalSummarizer,
        corruption,
        args: Namespace,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = Parameter(torch.empty(hidden_channels, hidden_channels))
        self.reset_parameters()
        self.K = args.K
        self.cluster_temp = args.clustertemp
        self.init = torch.rand(self.K, hidden_channels)

    def reset_parameters(self) -> None:
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        communities: list[set[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(features, edge_index, communities)
        embeddings = (
            torch.diag(1.0 / torch.norm(embeddings, p=2, dim=1))
            @ embeddings
        )
        community_tensors = [
            torch.tensor(list(community), dtype=torch.long)
            for community in communities
        ]
        centers = torch.stack(
            [
                torch.mean(
                    embeddings.index_select(0, community_tensor), dim=0
                )
                for community_tensor in community_tensors
            ],
            dim=0,
        )
        distances = embeddings @ centers.t()
        assignments = torch.softmax(self.cluster_temp * distances, dim=1)
        return embeddings, centers, assignments, distances

    @staticmethod
    def modularity(
        centers: torch.Tensor,
        assignments: torch.Tensor,
        embeddings: torch.Tensor,
        distances: torch.Tensor,
        adjacency: torch.Tensor,
        modularity_matrix: torch.Tensor,
        args: Namespace,
    ) -> torch.Tensor:
        del centers, embeddings, distances, args
        adjacency_without_diagonal = adjacency.clone()
        adjacency_without_diagonal.fill_diagonal_(0)
        adjacency_sum = adjacency_without_diagonal.sum()
        if adjacency_sum == 0:
            return torch.tensor(0.0)
        score = (
            assignments.t() @ modularity_matrix @ assignments
        ).trace()
        return -(score / adjacency_sum)


def make_modularity_matrix(adjacency: torch.Tensor) -> torch.Tensor:
    no_self_loops = adjacency * (
        torch.ones(adjacency.shape[0], adjacency.shape[0])
        - torch.eye(adjacency.shape[0])
    )
    degrees = no_self_loops.sum(dim=0).unsqueeze(1)
    return no_self_loops - degrees @ degrees.t() / no_self_loops.sum()


def select_original_communities(graph: nx.Graph) -> list[set[int]]:
    communities = nx.community.louvain_communities(
        graph,
        resolution=0.3,
        threshold=1e-9,
        seed=123,
    )
    sizes = np.asarray([len(community) for community in communities])
    threshold = sizes.mean() + 0.5 * sizes.std()
    return [
        community
        for community in communities
        if len(community) > threshold
    ]


def train_original_embeddings(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    adjacency: torch.Tensor,
    communities: list[set[int]],
    args: Namespace,
) -> tuple[np.ndarray, int]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.K = len(communities)
    model = OriginalDeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=OriginalEncoder(features.shape[1], args.hidden),
        summary=OriginalSummarizer(),
        corruption=original_corruption,
        args=args,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-3
    )
    modularity_matrix = make_modularity_matrix(adjacency)
    minimum_loss = float("inf")
    stale_epochs = 0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, 301):
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
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:3d}/300: loss={loss_value:.8f}")
        if stale_epochs >= 200:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeddings, _, _, _ = model(features, edge_index, communities)
    return embeddings.cpu().numpy(), best_epoch


def plot_original_tsne(
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
        title="Citeseer class",
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )
    ax.set_title(
        "Citeseer node embeddings — t-SNE\n"
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
        "./", "citeseer", "tensor", "npy", "npy", False, False, False
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
    output = Path("visualizations") / "citeseer_original_tsne.png"
    plot_original_tsne(
        embeddings, labels, len(communities), best_epoch, output
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Saved plot: {output.resolve()}")
    print(f"Saved coordinates: {output.with_suffix('.npz').resolve()}")


if __name__ == "__main__":
    main()
