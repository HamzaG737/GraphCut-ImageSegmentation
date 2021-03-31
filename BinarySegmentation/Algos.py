from queue import *
import numpy as np
import maxflow
from PIL import Image
import cv2


def BFS(ResGraph, V, s, t, parent):
    """
    Breadth first search algo.
    """
    q = Queue()
    VISITED = np.zeros(V, dtype=bool)
    q.put(s)
    VISITED[s] = True
    parent[s] = -1

    while not q.empty():
        p = q.get()
        for vertex in range(V):
            if (not VISITED[vertex]) and ResGraph[p][vertex] > 0:
                q.put(vertex)
                parent[vertex] = p
                VISITED[vertex] = True
    return VISITED[vertex]


def DFS(ResGraph, V, s, VISITED):
    """
    depth first search
    """
    current = [s]
    while current:
        v = current.pop()
        if not VISITED[v]:
            VISITED[v] = True
            current.extend([u for u in range(V) if ResGraph[v][u]])


def FordFulkerson(graph, s, t):
    print("Running Ford-Fulkerson algorithm")
    ResGraph = graph.copy()
    V = len(graph)
    parent = np.zeros(V, dtype="int32")

    while BFS(ResGraph, V, s, t, parent):
        pathFlow = float("inf")
        v = t
        while v != s:
            u = parent[v]
            pathFlow = min(pathFlow, ResGraph[u][v])
            v = parent[v]

        v = t
        while v != s:
            u = parent[v]
            ResGraph[u][v] -= pathFlow
            ResGraph[v][u] += pathFlow
            v = parent[v]

    VISITED = np.zeros(V, dtype=bool)
    DFS(ResGraph, V, s, VISITED)

    all_cuts = []

    for i in range(V):
        for j in range(V):
            if VISITED[i] and not VISITED[j] and graph[i][j]:
                all_cuts.append((i, j))
    return all_cuts


def boykov_kolmog(img_path, lbda, sigma, fore_grnd_sample, back_grnd_sample):
    """
    Implements Kolmogorov Boykov graph cut algorithm for image segmentation
    params:
    img_path : path to the input image
    lbda : hyperparameter of the cost function, defines similarity between pixels
    sigma : hyperparameter of the cost function, decay parameter.
    fore_grnd_sample : bounding box of the manually selected foreground area
    back_grnd_sample : bounding box of the manually selected background area
    """
    img = Image.open(img_path).convert("L")
    img_foreground = img.crop(fore_grnd_sample)
    img_background = img.crop(back_grnd_sample)
    img, img_foreground, img_background = (
        np.array(img),
        np.array(img_foreground),
        np.array(img_background),
    )
    fore_mean = np.mean(
        cv2.calcHist([img_foreground], [0], None, [256], [0, 256])
    )
    back_mean = np.mean(
        cv2.calcHist([img_background], [0], None, [256], [0, 256])
    )

    # initalizing foreground and background probabilities
    Foreground = np.ones(img.shape)
    Background = np.ones(img.shape)
    img_vec = img.reshape(-1, 1)
    H, W = img.shape[:2]

    # Initialize Graph
    graph = maxflow.Graph[int](H, W)
    tree = maxflow.Graph[int]()

    # Construct Trees
    nodes, nodeids = graph.add_nodes(H * W), tree.add_grid_nodes(img.shape)
    tree.add_grid_edges(nodeids, 0), tree.add_grid_tedges(
        nodeids, img, 255 - img
    )
    gr = tree.maxflow()
    segments = tree.get_grid_segments(nodeids)

    for i in range(H):
        for j in range(W):
            Foreground[i, j] = -np.log(
                abs(img[i, j] - fore_mean)
                / (abs(img[i, j] - fore_mean) + abs(img[i, j] - back_mean))
            )
            Background[i, j] = -np.log(
                abs(img[i, j] - back_mean)
                / (abs(img[i, j] - back_mean) + abs(img[i, j] - fore_mean))
            )
    Foreground = Foreground.reshape(-1, 1)
    Background = Background.reshape(-1, 1)

    # Normalizing
    for i in range(img_vec.shape[0]):
        img_vec[i] = img_vec[i] / np.linalg.norm(img_vec[i])

    for i in range(H * W):
        ws = Foreground[i] / (
            Foreground[i] + Background[i]
        )  # Calculating source weight
        wt = Background[i] / (
            Foreground[i] + Background[i]
        )  # Calculating sink weight
        graph.add_tedge(i, ws[0], wt)

        # Dealing with pixels on the border of the image
        if i % W != 0:
            w = lbda * np.exp(-(abs(img_vec[i] - img_vec[i - 1]) ** 2) / sigma)
            graph.add_edge(i, i - 1, w[0], lbda - w[0])

        if (i + 1) % W != 0:
            w = lbda * np.exp(-(abs(img_vec[i] - img_vec[i + 1]) ** 2) / sigma)
            graph.add_edge(i, i + 1, w[0], lbda - w[0])
        if i // W != 0:
            w = lbda * np.exp(-(abs(img_vec[i] - img_vec[i - W]) ** 2) / sigma)
            graph.add_edge(i, i - W, w[0], lbda - w[0])
        if i // W != H - 1:
            w = lbda * np.exp(-(abs(img_vec[i] - img_vec[i + W]) ** 2) / sigma)
            graph.add_edge(i, i + W, w[0], lbda - w[0])

    print("Maximum Flow: {}".format(gr))

    # Get binary labels and return mask
    segments_ = np.zeros(nodes.shape)
    for i in range(len(nodes)):
        segments_[i] = graph.get_segment(nodes[i])  # Get binary classification
    segments_ = segments_.reshape(img.shape[0], img.shape[1])
    mask = 255 * np.ones((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if segments[i, j] == False:
                mask[i, j] = 0
    return mask