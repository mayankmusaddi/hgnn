import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pc import Transform


cd = pd.read_csv('seqfish_plus/ge_withlabels.csv')
dt = {'Astrocyte' : 0, 'Choroid Plexus' : 1, 'Endothelial' : 2, 'Ependymal' : 3, 'Excitatory neuron' : 4, 'Interneuron' : 5, 'Microglia' : 6, 'Neural Stem' : 7, 'Neuroblast' : 8, 'Oligodendrocyte' : 9 }
labels = list(cd['labels'])
t = [dt[x] for x in labels]

pc = Transform(0)

X = np.load('embeddings.npy')
# X = np.zeros((913,4))
print(X.shape)
print(X[470])

embeddings2d = TSNE(n_components=2, metric=pc.geodesic).fit_transform(X)

# # Create DF
embeddingsdf = pd.DataFrame()# Add game names
# embeddingsdf['game'] = gameslist# Add x coordinate
embeddingsdf['x'] = embeddings2d[:,0]# Add y coordinate
embeddingsdf['y'] = embeddings2d[:,1]# Check
embeddingsdf.head()

# Set figsize
fig, ax = plt.subplots(figsize=(10,8))# Scatter points, set alpha low to make points translucent
ax.scatter(embeddingsdf.x, embeddingsdf.y, alpha=.5, c=t)
# ax.scatter(X[:,2], X[:,10], alpha=.5, c=t)
plt.title('Scatter plot of games using t-SNE')
plt.show()
