import cv2
import numpy as np


class GraphEmbedding:
    def __init__(self, path_img=None, array_input=None, sigma=20, resize=30):

        self.resizing_factor = resize
        if path_img == None:
            self.img_array = array_input
            self.width, self.height = (
                self.img_array.shape[0],
                self.img_array.shape[1],
            )
        else:
            self.img_array = self.OpenImg(path_img)
            self.width = self.height = self.resizing_factor
        self.embeddings_matrix = np.zeros(
            (self.height * self.width + 2, self.height * self.width + 2)
        )
        self.sigma = sigma

    def OpenImg(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.resizing_factor, self.resizing_factor))

        return image

    def compute_weight(self, pixel1, pixel2):
        penalty = 100 * np.exp(
            (-((pixel1 - pixel2) ** 2)) / (2 * (self.sigma ** 2))
        )
        return penalty

    def compute_edges(self):
        self.max_capacity = -np.inf
        for i in range(self.height):
            for j in range(self.width):
                l = i * self.width + j
                if i < self.height - 1:
                    k = (i + 1) * self.width + j
                    self.embeddings_matrix[k, l] = self.compute_weight(
                        self.img_array[i, j], self.img_array[i + 1, j]
                    )
                    self.embeddings_matrix[l, k] = self.embeddings_matrix[k, l]
                    self.max_capacity = max(
                        self.max_capacity, self.embeddings_matrix[k, l]
                    )
                if j < self.width - 1:
                    k = i * self.width + j + 1
                    self.embeddings_matrix[k, l] = self.compute_weight(
                        self.img_array[i, j], self.img_array[i, j + 1]
                    )
                    self.embeddings_matrix[l, k] = self.embeddings_matrix[k, l]
                    self.max_capacity = max(
                        self.max_capacity, self.embeddings_matrix[k, l]
                    )

    def compute_edges_source_sink(self, clusters_centers):

        for i in range(self.height):
            for j in range(self.width):
                l = i * self.width + j
                self.embeddings_matrix[-2][l] = self.compute_weight(
                    self.img_array[i, j], clusters_centers[0]
                )
        for i in range(self.height):
            for j in range(self.width):
                l = i * self.width + j
                self.embeddings_matrix[l][-1] = self.compute_weight(
                    self.img_array[i, j], clusters_centers[1]
                )
        """
        for i,j in seeds_fore :
            l = i * self.width + j 
            self.embeddings_matrix[-2][l] = self.max_capacity ## source index is img_size + 1 

        for i,j in seeds_back :
            l = i * self.width + j 
            self.embeddings_matrix[l][-1] = self.max_capacity ## sink index is img_size + 2
        """

    def compute_graph(self, clusters_centers):
        self.compute_edges()
        self.compute_edges_source_sink(clusters_centers)
