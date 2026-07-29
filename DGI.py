import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn.inits import reset, uniform
from sklearn.metrics.pairwise import euclidean_distances

EPS = 1e-15

class DeepGraphInfomax(torch.nn.Module):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """

    def __init__(self, hidden_channels, encoder, summary, corruption, args, cluster):
        super(DeepGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.reset_parameters()
        self.K = args.K
        self.cluster_temp = args.clustertemp
        self.init = torch.rand(self.K,hidden_channels)
        self.cluster = cluster

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def forward(self, *args, **kwargs):
        pos_z = self.encoder(*args, **kwargs)  # GCN学习节点表示
        pos_z = F.normalize(pos_z, p=2, dim=1, eps=1e-12)  # Stable equivalent L2 normalization.

        community_tensors = [torch.tensor(list(comm), dtype=torch.long) for comm in args[2]]
        center = [torch.mean(pos_z.index_select(0, comm_tensor), dim=0) for comm_tensor in community_tensors]
        mu = torch.stack(center, dim=0)

        dist = pos_z @ mu.t()
        r = torch.softmax(self.cluster_temp * dist, 1)
        return pos_z, mu, r, dist

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutal information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss #+ modularity

    def comm_loss(self,pos_z,mu):
        return -torch.log(self.discriminate(pos_z,self.summary(mu), sigmoid=True) + EPS).mean()

    def modularity(self, mu, r, embeds, dist, bin_adj, mod, args):
        # bin_adj_nodiag = bin_adj * (torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
        # loss = (1. / bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()
        device = bin_adj.device  # 确保计算在相同设备上
        bin_adj_nodiag = bin_adj.clone()
        bin_adj_nodiag.fill_diagonal_(0)  # 直接将对角线设置为 0，避免额外的矩阵计算
        adj_sum = bin_adj_nodiag.sum()

        if adj_sum == 0:
            return 0  # 防止除以零

        loss = (1. / adj_sum) * (r.t() @ mod @ r).trace()
        return -loss

    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
