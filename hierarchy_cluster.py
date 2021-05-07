import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering

cd = pd.read_csv('seqfish_plus/ge_withlabels.csv')
dt = {'Astrocyte' : 0, 'Choroid Plexus' : 1, 'Endothelial' : 2, 'Ependymal' : 3, 'Excitatory neuron' : 4, 'Interneuron' : 5, 'Microglia' : 6, 'Neural Stem' : 7, 'Neuroblast' : 8, 'Oligodendrocyte' : 9 }
ct = dict()
for i in dt:
	ct[dt[i]]=i
print(ct)
labels = np.array(cd['labels'])
t = np.array([dt[x] for x in labels])
print(t.shape, len(labels))

X = np.load('/home/anant/precog/hyp/Hyperbolic-GNNs/embeddings/embeddings.npy')

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Cell Dendograms")
dend = shc.dendrogram(shc.linkage(X, method='ward'))
print(t)
plt.axhline(linestyle='--', y=2.85)
plt.axhline(linestyle='--', y=5.4) 
# print(len(set(dend['ivl'])))
plt.show()

# cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
# cluster.fit_predict(X)

# print(cluster.labels_)

# plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
# plt.show()