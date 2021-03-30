
import maxflow as mf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import cluster
import argparse

from utils import *
#utils


def calculate_energy(img, DataP, img_labels):
    '''Calculates Energy of image.
       img: is input array'''

    E_data = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            E_data += DataCost(img_labels[i][j], DataP, j, i)
    
    E_smooth = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            ns = give_neighbours(img_labels, j, i)
            E_smooth += sum([SmoothCost(v, img_labels[i][j]) for v in ns])

    return E_data + E_smooth

     
def SmoothCost(l1, l2):
    #smoothness term
    if l1 == l2:
      return 0
    else:
      return 3
    
    
def DataCost(label, DataP, x, y):
	#Data cost term
    return -np.log(DataP[y,x,label])
    

def give_neighbours(image, x, y):
    '''Returns a list of all neighbour intensities'''
    ns = []
    for a,b in zip([1,0,-1,0],[0,1,0,-1]):
        if (x+a<len(image[0]) and x+a>=0) and (y+b<len(image) and y+b>=0):
            ns.append(image[y+b][x+a])
    return ns 


def return_mapping_of_image(image, alpha, beta):
    mapping = {}
    revmap = {}

    #add pixels and add them to maps
    map_parameter = 0
    for y in range(len(image)):
        for x in range(len(image[0])):
            #extract pixel which have the wanted label
            if image[y,x] == alpha or image[y,x] == beta:
                mapping[map_parameter] = (y,x)
                revmap[(y,x)] = map_parameter
                map_parameter += 1
    
    return mapping, revmap

def alpha_beta_swap_new(alpha, beta, img_orig, img_labels, DataP):
    """ 
    Applies alpha-beta move algorithm

    Args:
	    alpha, beta: 2 labels to be swapped  
	    img_orig: original grayscale image,  shape = (N x P)
	    img_labels: initialized labeled image,   shape = (N x P)
	    DataP: Probability distribution ,   shape = (N x P x n_labels) 
    """

    #extract position of alpha or beta pixels to mapping 
    map, revmap = return_mapping_of_image(img_labels, alpha, beta)
    
    #graph of maxflow 
    graph_mf = mf.Graph[float](len(map))
    #add nodes
    nodes = graph_mf.add_nodes(len(map))
            
    #add n-link edges
    weight = SmoothCost(alpha, beta)
    for i in range(0,len(map)):
        y,x = map[i]
        #top, left, bottom, right
        for a,b in zip([1,0,-1,0],[0,1,0,-1]):
            if (y+b, x+a) in revmap:
                graph_mf.add_edge(i,revmap[(y+b,x+a)], weight, 0)
   
    #add all the terminal edges
    for i in range(0,len(map)):
        y,x = map[i]
        #find neighbours
        neighbours = give_neighbours(img_labels, x, y)
        #consider only neighbours which are not having alpha or beta label
        fil_neigh = list(filter(lambda i: i!=alpha and i!=beta, neighbours))
        #calculation of weight
        t_weight_alpha = sum([SmoothCost(alpha,v) for v in fil_neigh]) + DataCost(alpha, DataP, x, y)
        t_weight_beta = sum([SmoothCost(beta,v) for v in fil_neigh]) + DataCost(beta, DataP, x, y)
        graph_mf.add_tedge(nodes[i], t_weight_alpha, t_weight_beta)

    #calculating flow
    flow = graph_mf.maxflow()
    res = [graph_mf.get_segment(nodes[i]) for i in range(0, len(nodes))]
    
    #depending on cut assign new label
    for i in range(0, len(res)):
        y, x = map[i] 
        if res[i] == 1:
            img_labels[y][x] = alpha 
        else:
            img_labels[y][x] = beta
    
    return img_labels


def swap_minimization(img, img_labels, DataP, n_cycles):

    best_energy = calculate_energy(img, DataP, img_labels)
    print("initial energy ", best_energy)
    labels = np.unique(img_labels)
    #do iteration of all pairs a few times
    for u in range(n_cycles):
        print('cycle : ',u)
        #iterate over all pairs of labels 
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):

                #computing intensive swapping and graph cutting part
                aux  = alpha_beta_swap_new(labels[i],labels[j], img, img_labels, DataP) 
                new_energy = calculate_energy(img, DataP, aux)
                #print('new energy ', new_energy)
                if  new_energy < best_energy:
                    img_labels = aux  
                    best_energy = new_energy
                    print("new best energy ", best_energy)
                          
    return img_labels


def main(args): 

	#load image
	n_labels = args.n_labels
	img_path =  args.ImgPath 

	img_orig = Image.open(img_path).convert('L')
	img_orig = img_orig.resize((500,334))
	img_orig = np.asarray(img_orig)

	#get labels and reshaped to the image shape
	labels = km_clust(img_orig, n_clusters=n_labels)
	labels.shape = img_orig.shape

	#define (interactively) bounds of regions defining the different objects
	coords = [
	          (100,250,50,100),
	          (150,250,160,200),
	          (200,250,260,300),
	          (150,200,380,410),
	          (0,50,0,200),
	]

	#associate labels to each region
	coords_dic = {}
	for coord in coords:
	  l = getLabel(labels, coord)
	  coords_dic[l] = coord

	#Create the data probability array
	n,p = img_orig.shape
	data_prob = np.zeros((n,p,n_labels))

	for label in np.arange(n_labels):
	  data_prob[:,:,label] = getProba(img_orig, coords_dic[label])

	n_cycles = args.n_cycles
	#Run the alpha beta swap minimization to get the segmented image
	labeled_img = swap_minimization(img_orig, labels, data_prob, n_cycles) 

	#save
	plt.imsave('images/segmented.png', labeled_img)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="args for graph cut ")
    parser.add_argument(
        "--n_labels",
        type=int,
        default=3,
        help="number of labels in the image ",
    )

    parser.add_argument(
        "--n_cycles",
        type=int,
        default=3,
        help="number of cycles for the move algorithm",
    )

    parser.add_argument(
        "--ImgPath",
        type=str,
        required=True,
        help="path to image ",
    )
    args = parser.parse_args()
    main(args)