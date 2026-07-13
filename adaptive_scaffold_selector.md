# Adaptive Scaffold Selector

This branch adds a lightweight pre-training selector that chooses between Louvain
and Leiden before running the full 15-run consensus scaffold and GNN training.

## Motivation

The full experiment can run both Louvain and Leiden, then compare final model
metrics. That is expensive and also makes the algorithm choice happen after the
training result is already known. The selector instead estimates which community
detection method is likely to provide the more reliable scaffold before the GNN
training step.

The selector is label-free and does not use final clustering metrics such as
NMI, accuracy, F1, ARI, DBI, or modularity against the dataset labels. It only
uses cheap structural signals from probe community-detection runs.

## Selection Procedure

For each dataset:

1. Run Louvain for 3 probe seeds.
2. Run Leiden for 3 probe seeds.
3. Score each method using scaffold reliability signals.
4. Choose the higher-scoring method.
5. Run the normal 15-run consensus scaffold only for the chosen method.
6. Train and evaluate the model as before.

## Selector Signals

### 1. Partition Stability

Partition stability measures whether the same algorithm produces similar
communities across different random seeds.

For three probe partitions, the code computes pairwise NMI:

```text
NMI(run_1, run_2)
NMI(run_1, run_3)
NMI(run_2, run_3)
```

The stability score is the average of these pairwise values. A higher value means
the algorithm is less seed-sensitive on the current graph.

### 2. Usable K Stability

The model filters detected communities before training:

```text
threshold = mean(community_size) + 0.5 * std(community_size)
selected communities = communities larger than threshold
K = number of selected communities
```

For each probe run, the selector computes this usable `K`. Then it measures how
stable `K` is across the three probe runs:

```text
k_stability = 1 / (1 + std(K values))
```

This favors algorithms that provide a consistent number of usable scaffold
communities for the GNN.

### 3. Community Balance

Community balance penalizes two undesirable scaffold shapes:

- one giant community containing too much of the graph
- many tiny communities that are unlikely to be useful as stable scaffold groups

For each probe run:

```text
largest_ratio = largest community size / number of nodes
tiny_ratio = tiny communities / total communities
balance = 1 - 0.5 * largest_ratio - 0.5 * tiny_ratio
```

The balance score is clamped to `[0, 1]` and averaged across probe runs.

## Final Score

The selector combines the three signals as:

```text
score = 0.60 * partition_stability
      + 0.25 * usable_K_stability
      + 0.15 * community_balance
```

The selected algorithm is printed for each dataset with the reasoning values:

```text
score
partition stability
K stability
community balance
probe K values
```

These selector diagnostics are only printed to the console. They are not written
to the results CSV.

## CSV Output

The CSV remains focused on model results only:

```text
dataset,p_feat,algorithm,K,nmi,accuracy,f1,ari,dbi,modularity,elapsed_seconds
```

The `algorithm` column records the algorithm selected by the adaptive selector.
No selector-internal values are added to the CSV.
