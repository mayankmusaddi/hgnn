import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import h5py

file1 = "osmfish/osmFISH_SScortex_mouse_all_cells.loom"
f = h5py.File(file1,mode = 'r')
meta = f['col_attrs']
cell_types = np.asarray(meta['ClusterID'])
cell_names = np.asarray(meta['ClusterName'])

X = np.load('/home/anant/precog/hyp/Hyperbolic-GNNs/embeddings/embeddings.npy')
# X = np.zeros((913,4))

print(X.shape)
print(X[692])

embeddings2d = TSNE(n_components=2).fit_transform(X)

# # Create DF
embeddingsdf = pd.DataFrame()# Add game names
# embeddingsdf['game'] = gameslist# Add x coordinate
embeddingsdf['x'] = embeddings2d[:,0]# Add y coordinate
embeddingsdf['y'] = embeddings2d[:,1]# Check
embeddingsdf.head()
yy = set(cell_types)
print(yy)
# Set figsize
fig, ax = plt.subplots(figsize=(10,8))# Scatter points, set alpha low to make points translucent

for g in np.unique(cell_types):
    i = np.where(cell_types == g)
    print(i, g)
    ax.scatter(embeddings2d[i,0], embeddings2d[i,1], label = cell_names[np.where(cell_types==g)][0],alpha = 0.7)
    ax.legend()
# ax.scatter(embeddingsdf.x, embeddingsdf.y, alpha=.5, c=t)
# ax.scatter(X[:,2], X[:,10], alpha=.5, c=t)
plt.title('t-SNE Scatter-Plot')

plt.show()
