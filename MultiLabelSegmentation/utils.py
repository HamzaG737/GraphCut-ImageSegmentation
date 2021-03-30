import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import cluster

def km_clust(array, n_clusters):
    
    # Create a line array, the lazy way
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = cluster.KMeans(n_clusters=n_clusters, n_init=4)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the label of each point
    labels = k_m.labels_
    return labels

def getLabel(labels, coord):
  region = labels[coord[0]:coord[1], coord[2]:coord[3]]
  values, counts = np.unique(region, return_counts=True)
  return values[np.argmax(counts)]

def getProba(img, coord):
  region = img[coord[0]:coord[1], coord[2]:coord[3]]
  hist = cv2.calcHist( [region], [0], None, [256],  [0,256])
  hist = hist / hist.sum()
  regAlpha = 0.01
  hist = regAlpha/hist.size + (1.0-regAlpha)*hist
  hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

  prob = cv2.calcBackProject([img], [0], hist, [0,256], 1.0)
  prob = prob.astype(float)/255.
  prob = prob + 1e-6

  return prob