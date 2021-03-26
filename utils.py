from queue import *
from sklearn.cluster import KMeans
import numpy as np


def get_neighbours(ver, height, width):
    neighbours = []
    i, j = ver[0], ver[1]
    if i < height - 1:
        neighbours.append((i + 1, j))
    if i > 0:
        neighbours.append((i - 1, j))
    if j > 0:
        neighbours.append((i, j - 1))
    if j < width - 1:
        neighbours.append((i, j + 1))
    return neighbours


def compute_mask(cut_edges, height, width, source, sink):
    edges_left = [cut[0] for cut in cut_edges]
    edges_right = [cut[1] for cut in cut_edges]

    tab_vertices_left = [
        (vertice // width, vertice % width)
        for vertice in edges_left
        if vertice not in [source, sink]
    ]
    tab_vertices_right = [
        (vertice // width, vertice % width)
        for vertice in edges_right
        if vertice not in [source, sink]
    ]
    q = Queue()
    for vertice in tab_vertices_left:
        q.put(vertice)

    while not q.empty():
        vertice = q.get()
        neighbours = get_neighbours(vertice, height, width)
        for neighbour in neighbours:
            if neighbour in tab_vertices_right + tab_vertices_left:
                continue
            tab_vertices_left.append(neighbour)
            q.put(neighbour)
    mask = np.zeros((height, width))
    for pair in tab_vertices_left:
        mask[pair[0], pair[1]] = 1
    return mask


def get_clusters(gr):
    X = np.zeros((gr.img_array.size, 1))
    for i in range(gr.img_array.shape[0]):
        for j in range(gr.img_array.shape[1]):
            X[i * gr.img_array.shape[1] + j] = gr.img_array[i, j]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    return kmeans.cluster_centers_
