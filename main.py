from model import Encoder, corruption, Summarizer, cluster_net
from utils.load_data import load_data
from DGI import DeepGraphInfomax

import evaluation
import time

from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
from clusteval import clusteval
from sklearn import cluster

from datetime import datetime, timezone, timedelta
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import torch
import os

from openpyxl import load_workbook
from openpyxl import Workbook


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate.')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default="cora",
                    help='which network to load')
parser.add_argument('--color', type=str, default='r-',
                    help='color line')
parser.add_argument('--K', type=int, default=7,
                    help='How many partitions')
parser.add_argument('--clustertemp', type=float, default=30,
                    help='how hard to make the softmax for the cluster assignments')
parser.add_argument('--train_iters', type=int, default=1001,
                    help='number of training iterations')
parser.add_argument('--num_cluster_iter', type=int, default=1,
                    help='number of iterations for clustering')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')


args = parser.parse_args()


def make_modularity_matrix(adj):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod


def count(label):
    cnt = [0] * 20
    for i in label:
        cnt[i] += 1
    print(cnt)


def result(graph, pred, labels):
    pred_np = pred.numpy()
    nmi = evaluation.NMI_helper(pred_np, labels)
    ac = evaluation.matched_ac(pred_np, labels)
    f1 = evaluation.cal_F_score(pred_np, labels)[0]
    ari = evaluation.adjusted_rand_score(pred_np, labels)
    q = evaluation.compute_modularity(graph, pred)
    return nmi, ac, f1, ari,q


def train():
    model.train()
    optimizer.zero_grad()
    pos_z, mu, r, dist = model(feat, edge, selected_communities)
    modularity_loss = model.modularity(mu, r, pos_z, dist, adj, test_object, args)
    loss = b * modularity_loss
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model):
    model.eval()

    with torch.no_grad():
        node_emb, mu, r, _ = model(feat, edge, selected_communities)

    r_assign = r.argmax(dim=1)
    print('label is:')
    count(label)
    print('result of r_assign is:')
    count(r_assign)
    r_nmi, r_ac, r_f1, r_ari, q = result(graph, r_assign, label)
    DBI = davies_bouldin_score(node_emb, r_assign)
    print("New Center Metrics: ", r_nmi, r_ac, r_f1, r_ari, DBI, q)
    return node_emb, r_assign, r_nmi, r_ac, r_f1, r_ari, DBI, q

device = torch.device('cpu')
a = 0.0
b = 0.001
c = 0.0

file_name = "result.csv"

start_time = time.perf_counter()

print("****************************", args.dataset, "dataset ******************************")
file = open(file_name, "a+")
print("****************************", args.dataset, "dataset ******************************", file=file)
file.close()

data = load_data("./", args.dataset,
                 "tensor", "npy", "npy",
                 False, False, False, None)
feat = data.feature.type(torch.float32)

A = data.adj  # numpy
adj = torch.tensor(A).type(torch.float32)
indices = np.where(A == 1)
indices_array = np.array(indices)
edge = torch.tensor(indices_array)

test_object = make_modularity_matrix(adj)

label = data.label

num_features = feat.shape[1]

graph = nx.from_numpy_array(A)
structure_community = nx.community.louvain_communities(graph, resolution=0.3, threshold=1e-09, seed=123)  # re=0.3

structure_community_node_number = [len(i) for i in structure_community]
mean_size = np.mean(structure_community_node_number)
std_deviation = np.std(structure_community_node_number)
threshold = mean_size + 0.5*std_deviation

selected_communities = [community for community in structure_community if len(community) > threshold]

K = len(selected_communities)
print(f"The Number of Selected Structural Communities: {K}", end='')
print(*[len(i) for i in selected_communities], sep=' ')

args.K = K
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

hidden_size = args.hidden
model = DeepGraphInfomax(
    hidden_channels=hidden_size, encoder=Encoder(num_features, hidden_size),
    summary=Summarizer(),
    corruption=corruption,
    args=args,
    cluster=cluster_net).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)

loss_para = f'{a}_{b}_{c} loss'

max_nmi = 0
max_ac = 0
max_ari = 0
max_f1 = 0
min_dbi = 3
max_q = 0

print(f"Start training {a}_{b}_{c} loss para!!!=================================")
stop_cnt = 0
best_idx = 0
patience = 200
min_loss = 1e9
real_epoch = 0

for epoch in range(1, 301):
    loss = train()
    if epoch % 2 == 0 and epoch > 0:
        print('epoch = {}'.format(epoch))
        final_z, final_r, tmp_max_nmi, tmp_max_ac, tmp_max_f1, tmp_max_ari, tmp_max_dbi, tmp_max_q = test(model)

        max_nmi = max(max_nmi, tmp_max_nmi)
        max_ac = max(max_ac, tmp_max_ac)
        max_f1 = max(max_f1, tmp_max_f1)
        max_ari = max(max_ari, tmp_max_ari)
        min_dbi = min(min_dbi, tmp_max_dbi)
        max_q = max(max_q, tmp_max_q)
        print("----------------------------------------------------------")
    if loss < min_loss:
        min_loss = loss
        best_idx = epoch
        stop_cnt = 0
        torch.save(model.state_dict(), 'best_model.pkl')
    else:
        stop_cnt += 1
    if stop_cnt >= patience:
        real_epoch = epoch
        break

print('Loading {}th epoch'.format(best_idx))
model.load_state_dict(torch.load('best_model.pkl'))
print('Start testing !!!')
test(model)
print("max nmi为", max_nmi)
print("max ac为:", max_ac)
print("max f1为:", max_f1)
print("max ari为:", max_ari)
print("min dbi为", min_dbi)
print("max q:", max_q)


file = open(file_name, "a+")
print(f"The result of {a}_{b}_{c} loss para:-----------------", file=file)
print("\tmin DBI:", min_dbi,
      "\n\tmax Q:", max_q,
      "\n\tmax NMI:", max_nmi,
      "\n\tmax ACC:", max_ac,
      "\n\tmax F1:", max_f1,
      "\n\tmax ARI:", max_ari,
      file=file)
file.close()
end_time = time.perf_counter()
running_time = end_time - start_time
print(f"The running time:{running_time}s")
file = open(file_name, "a+")
print(f"The running time:{running_time}s", file=file)