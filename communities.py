# for finding optimal k
from gap_statistic.optimalK import OptimalK

# for DNGR embeddings
import relegy.embeddings as rle

# for clustering
from scipy.cluster.hierarchy import linkage, fcluster
from time import time as measure_time

import os
import numpy as np
import networkx as nx


# complete cosine works best
def hierarchical_clustering(X, k, link="ward"):
    print(f"Clustering with k={k}")
    link = linkage(X, method=link, metric="euclidean")
    clusters = fcluster(link, k, criterion="maxclust")
    return clusters


def get_centroids_and_clustering(X, clusts):
    centroids = []

    for i in range(1, max(clusts) + 1):
        centroids.append(np.mean(X[clusts == i], axis=0))

    return [list(centroid) for centroid in centroids], [clust - 1 for clust in clusts]


def gap_stat(X, k):
    return get_centroids_and_clustering(X, hierarchical_clustering(X, k))


def find_optimal_k(embeddings):
    optimal_k = OptimalK(clusterer=gap_stat, parallel_backend="joblib", n_jobs=-1)
    return optimal_k(embeddings, cluster_array=np.arange(1, 20))


def cluster_nodes(G, k=None):
    n = G.number_of_nodes()
    # embedding size -- dngr paper recommends min n, 2 log(n)
    d = round(min(n, 2 * np.log2(n)))

    embeddings = rle.DNGR.fast_embed(
        G, d=d, n_layers=2, n_hid=[n, d], dropout=0.05, num_iter=200
    )

    if k is None:
        k = find_optimal_k(embeddings)

    return hierarchical_clustering(embeddings, k)


def main():
    graphs = os.listdir("Lab10/snars_competition/competition")
    execution_times = {}

    for graph in graphs:
        # if not graph.startswith("D1-U"):
        #     continue
        time_start = measure_time()

        adjecency_matrix = np.loadtxt(
            f"Lab10/snars_competition/competition/{graph}", delimiter=","
        )
        G = nx.from_numpy_array(adjecency_matrix)

        if (kstring := graph.split(".")[0].split("-")[1]) == "UNC":
            k = None
        else:
            k = int(kstring.split("=")[1])

        clustering = cluster_nodes(G, k=k)
        print(clustering)

        with open(f"Lab10/snars_competition/results/{graph}", "w") as f:
            for node, cluster in enumerate(clustering):
                f.write(f"{node+1}, {cluster}\n")

        time_end = measure_time()
        execution_times[graph] = time_end - time_start

    with open(f"Lab10/snars_competition/results/description.txt", "w") as f:
        f.write("Mikołaj Spytek, Mateusz Krzyziński\n")
        f.write("https://github.com/krzyzinskim/snars_competition\n")
        for graph, time in execution_times.items():
            f.write(f"{graph}, {time}\n")


if __name__ == "__main__":
    main()
