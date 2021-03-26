from Algos import FordFulkerson
from utils import compute_mask, get_clusters
from GraphEmbedding import GraphEmbedding
import argparse
import cv2
import matplotlib.pyplot as plt

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
    plt.imsave("mask.jpg", mask, cmap="gray")
    print("Saved image as", "ImgMask.jpg")


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