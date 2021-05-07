import h5py
import os
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

def main():
    file1 = "osmfish/osmFISH_SScortex_mouse_all_cells.loom"
    f = h5py.File(file1,mode = 'r')
    gene_expression =  np.asarray(f['matrix'])
    gene_expression = np.transpose(gene_expression)
    gene_sum = np.sum(gene_expression,axis = 1,keepdims = True)
    zero_mask = gene_sum != 0
    zero_mask = np.reshape(zero_mask,len(zero_mask))
    gene_expression = gene_expression[zero_mask,:]
    gene_sum = gene_sum[zero_mask,:]
    meta = f['col_attrs']
    genes = np.asarray(f['row_attrs']['Gene'])
    genes = [x.decode() for x in genes]
    x = np.asarray(meta['X'])[zero_mask]
    y = np.asarray(meta['Y'])[zero_mask]
    coordinates = np.stack((x,y),axis=1)
    plt.scatter(x,y,s = 1)
    cell_types = np.asarray(meta['ClusterID'])[zero_mask]
    cell_names = np.asarray(meta['ClusterName'])
    n_c = len(set(cell_types))

    print(gene_expression.shape)
    print(coordinates.shape)
    print(cell_types.shape)
    print(set(cell_types))
    # print(cell_names)
    print(gene_expression[0])
    # print(coordinates)
    print(cell_names[0])
    print(cell_types[0])

    # exit()
    # distance_list_list = []
    # print ('calculating distance matrix, it takes a while')
    # for j in range(coordinates.shape[0]):
    #     distance_list = []
    #     for i in range (coordinates.shape[0]):
    #         if i!=j:
    #             distance_list.append(np.linalg.norm(coordinates[j]-coordinates[i]))
    #     distance_list_list.append(distance_list)
    #     # break
    
    # np.save('osmfish/distance_array.npy',np.array(distance_list_list))
    # exit()
    
    ############ the distribution of distance
    from sklearn.metrics.pairwise import euclidean_distances
    distance_matrix = euclidean_distances(coordinates, coordinates)
    thresh_dm = np.zeros(distance_matrix.shape)
    thresh_dm[np.where(distance_matrix<140)] = 1
    print(np.where(thresh_dm==1)[0].shape)
    # for threshold in [100,140,180,210,220,260]: #range (210,211):#(100,400,40):
    #     num_big = np.where(distance_matrix<threshold & (distance_matrix>0))[0].shape[0]
    #     print (threshold,num_big,str(num_big/(6430*2)))
    # # exit()
    # distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
    # for i in range(distance_matrix_threshold_I.shape[0]):
    #     for j in range(distance_matrix_threshold_I.shape[1]):
    #         if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
    #                 distance_matrix_threshold_I[i,j] = 1
    np.save('osmfish/adjacency_matrix.npy',np.array(thresh_dm))
    # exit()

    print("here")
    import networkx as nx
    g = nx.Graph()
    
    # cd = pd.read_csv('seqfish_plus/ge_withlabels.csv')
    am = np.load('osmfish/adjacency_matrix.npy')
    # dt = {'Astrocyte' : 0, 'Choroid Plexus' : 1, 'Endothelial' : 2, 'Ependymal' : 3, 'Excitatory neuron' : 4, 'Interneuron' : 5, 'Microglia' : 6, 'Neural Stem' : 7, 'Neuroblast' : 8, 'Oligodendrocyte' : 9 }

    for i in range(gene_expression.shape[0]):
        row = np.array(gene_expression[i])
        g.add_node(i,features=row,label=cell_types[i])
    for i in range(am.shape[0]):
        for j in range(am.shape[1]):
            if am[i][j] == 1:
                g.add_edge(i,j)
    
    print(g.number_of_nodes())
    print(g.number_of_edges())

    nx.write_gpickle(g, 'gen.gpickle')

if __name__=='__main__':
    main()