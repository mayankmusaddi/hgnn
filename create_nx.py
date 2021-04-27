import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import euclidean_distances
####################  get the whole training dataset

current_path = os.path.abspath('.')
# cortex_svz_cellcentroids = pd.read_csv(current_path+'/seqfish_plus/cortex_svz_cellcentroids.csv')
# ############# get batch adjacent matrix
# cell_view_list = []
# for view_num in range(7):
#     cell_view = cortex_svz_cellcentroids[cortex_svz_cellcentroids['Field of View']==view_num]
#     cell_view_list.append(cell_view)

# ############ the distribution of distance
# distance_list_list = []
# distance_list_list_2 = []
# print ('calculating distance matrix, it takes a while')
# for view_num in range(7):
#     print (view_num)
#     cell_view = cell_view_list[view_num]
#     distance_list = []
#     for j in range(cell_view.shape[0]):
#         for i in range (cell_view.shape[0]):
#             if i!=j:
#                 distance_list.append(np.linalg.norm(cell_view.iloc[j][['X','Y']]-cell_view.iloc[i][['X','Y']]))
#     distance_list_list = distance_list_list + distance_list
#     distance_list_list_2.append(distance_list)

# # np.save(current_path+'/seqfish_plus/distance_array.npy',np.array(distance_list_list))
# ###try different distance threshold, so that on average, each cell has x neighbor cells, see Tab. S1 for results
# from scipy import sparse
# import pickle
# import spektral
# import scipy.linalg
# distance_array = np.array(distance_list_list)
# for threshold in [140]:#[100,140,180,210,220,260]:#range (210,211):#(100,400,40):
#     num_big = np.where(distance_array<threshold)[0].shape[0]
#     print (threshold,num_big,str(num_big/(913*2)))
#     distance_matrix_threshold_I_list = []
#     distance_matrix_threshold_W_list = []
#     from sklearn.metrics.pairwise import euclidean_distances
#     for view_num in range (7):
#         cell_view = cell_view_list[view_num]
#         distance_matrix = euclidean_distances(cell_view[['X','Y']], cell_view[['X','Y']])
#         distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
#         distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
#         for i in range(distance_matrix_threshold_I.shape[0]):
#             for j in range(distance_matrix_threshold_I.shape[1]):
#                 if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
#                     distance_matrix_threshold_I[i,j] = 1
#                     distance_matrix_threshold_W[i,j] = distance_matrix[i,j]
#         distance_matrix_threshold_I_list.append(distance_matrix_threshold_I)
#         distance_matrix_threshold_W_list.append(distance_matrix_threshold_W)
#     whole_distance_matrix_threshold_I = scipy.linalg.block_diag(distance_matrix_threshold_I_list[0],
#                                                                 distance_matrix_threshold_I_list[1],
#                                                                 distance_matrix_threshold_I_list[2],
#                                                                 distance_matrix_threshold_I_list[3],
#                                                                 distance_matrix_threshold_I_list[4],
#                                                                 distance_matrix_threshold_I_list[5],
#                                                                 distance_matrix_threshold_I_list[6])
# np.save(current_path+'/seqfish_plus/adjacency_matrix.npy',np.array(whole_distance_matrix_threshold_I))
# print(whole_distance_matrix_threshold_I)

import networkx as nx
g = nx.Graph()

cd = pd.read_csv('seqfish_plus/ge_withlabels.csv')
am = np.load('seqfish_plus/adjacency_matrix.npy')
dt = {'Astrocyte' : 0, 'Choroid Plexus' : 1, 'Endothelial' : 2, 'Ependymal' : 3, 'Excitatory neuron' : 4, 'Interneuron' : 5, 'Microglia' : 6, 'Neural Stem' : 7, 'Neuroblast' : 8, 'Oligodendrocyte' : 9 }

for i in range(cd.shape[0]):
    row = np.array(cd.iloc[i])
    g.add_node(i,features=row[:-1],label=dt[row[-1]])
for i in range(am.shape[0]):
    for j in range(am.shape[1]):
        if am[i][j] == 1:
            g.add_edge(i,j)

print(g.number_of_nodes())
print(g.number_of_edges())

nx.write_gpickle(g, "./general.gpickle")
nx.write_gpickle(g, 'gen.gpickle')
