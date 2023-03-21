import os
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from random import randint, sample
from scipy.linalg import subspace_angles
# from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform

WS_LOC = '../data/subspaces/goi5k/'


def get_subspace(folder: str = WS_LOC, index = None) -> np.ndarray:
    npy_subspaces = os.listdir(WS_LOC)
    if index is None:
        index = randint(0, len(npy_subspaces))
    S = np.load(os.path.join(WS_LOC, npy_subspaces[index]))
    return S


def sample_subspaces(folder: str = WS_LOC, count: int = 10) -> tuple:
    sample_subs = sample(population=os.listdir(WS_LOC),k=count)
    ids = [sub.split('.')[0] for sub in sample_subs]
    return ids, np.asarray([np.load(os.path.join(WS_LOC, sub)) for sub in sample_subs], dtype=np.float32)
#     return [np.load(os.path.join(WS_LOC, sub)) for sub in sample_subs]


def subspaces_similarity(S1, S2):
    canon_angles = subspace_angles(S1, S2)
    return np.average(np.square(np.cos(canon_angles)))


def condensed_dist(subs):
    n = subs.shape[0]
    dm = []
    for i in range(n):
        s_i = subs[i]
        dm.extend([1 - subspaces_similarity(S1=s_i, S2=subs[j]) for j in range(i+1, n)])
    return np.array(dm)


class Cluster:

    def __init__(self, label: int, subspace, sub_id: str):
        self.label = label
        self.size = 1
        self.subspaces = np.expand_dims(subspace, axis=0)
        self.rep = subspace
        self.ids = [sub_id]

    def __repr__(self):
        return f'Cluster-{self.label}'

    def __lt__(self, other):
        return True

    def __eq__(self, other):
        return True

    def distance(self, other):
        # other is subspace (300, 5)
        return 1 - subspaces_similarity(S1=self.rep, S2=other)

    def add(self, subspace, sub_id: str):
        self.subspaces = np.append(self.subspaces, np.expand_dims(subspace, axis=0), axis=0)
        self.size += 1
        self.compute_rep()
        self.ids.append(sub_id)

    def compute_rep(self):
        self.rep = np.mean(self.subspaces, axis=0, dtype=np.float64)

    def get_rep(self):
        return self.rep


class Index:

    def __init__(self, subspaces: np.ndarray):
        pass


if __name__ == '__main__':
    # sub_one = get_subspace()
    # sub_two = get_subspace()
    # sim = subspaces_similarity(S1=sub_one, S2=sub_two)
    # print(sim)
    count = 10
    ids, subs = sample_subspaces(count=count)
    dm = condensed_dist(subs)
    s_dm = squareform(dm)

    n_clusters = int(np.sqrt(count))
    affinity = 'precomputed'
    lnkg = 'complete'  # single or complete
    compute_full_tree = 'auto'
    if n_clusters is not None:
        distance_threshold = None
    else:
        distance_threshold = 0.85
    ac = AgglomerativeClustering(n_clusters=n_clusters,
                                 affinity=affinity,
                                 linkage=lnkg,
                                 compute_full_tree=compute_full_tree,
                                 distance_threshold=distance_threshold)
    ac.fit(s_dm)
    print(f'Number of clusters: {ac.n_clusters_}')

    labels = ac.labels_
    print(f'labels: {labels}')

    clusters = {}
    for i, s, l in zip(ids, subs, labels):
        if l not in clusters:
            clusters[l] = Cluster(label=l, subspace=s, sub_id=i)
        else:
            clusters[l].add(subspace=s, sub_id=i)

    print(clusters[0].distance(subs[-3]))