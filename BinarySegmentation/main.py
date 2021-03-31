from Algos import FordFulkerson
from utils import compute_mask, get_clusters
from GraphEmbedding import GraphEmbedding
import argparse
import cv2
import matplotlib.pyplot as plt
import os

sigma = 30


def main(args):
    graph = GraphEmbedding(
        path_img=args.ImgPath, sigma=sigma, resize=args.resize_factor
    )
    cluster_centers = get_clusters(graph)
    graph.compute_graph(cluster_centers)
    embeddings = graph.embeddings_matrix
    source = len(embeddings) - 2
    sink = len(embeddings) - 1

    cut_edges = FordFulkerson(embeddings, source, sink)
    mask = compute_mask(cut_edges, graph.height, graph.width, source, sink)
    mask_reshape = cv2.resize(mask, graph.original_size[::-1], 0, 0)
    path, file_ = os.path.split(args.ImgPath)
    filename, file_extension = os.path.splitext(file_)
    save_dir = os.path.join(path, filename + "-mask" + file_extension)

    plt.imsave(save_dir, mask_reshape, cmap="gray")
    print("Saved image in ", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for graph cut ")
    parser.add_argument(
        "--resize_factor",
        type=int,
        default=30,
        help="maximum shape of image ",
    )
    parser.add_argument(
        "--ImgPath",
        type=str,
        help="path to image ",
    )
    args = parser.parse_args()
    main(args)