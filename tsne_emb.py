import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


cd = pd.read_csv('seqfish_plus/ge_withlabels.csv')
dt = {'Astrocyte' : 0, 'Choroid Plexus' : 1, 'Endothelial' : 2, 'Ependymal' : 3, 'Excitatory neuron' : 4, 'Interneuron' : 5, 'Microglia' : 6, 'Neural Stem' : 7, 'Neuroblast' : 8, 'Oligodendrocyte' : 9 }
ct = dict()
for i in dt:
	ct[dt[i]]=i
print(ct)
labels = list(cd['labels'])
t = np.array([dt[x] for x in labels])


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
yy = set(t)
print(yy)
# Set figsize
fig, ax = plt.subplots(figsize=(10,8))# Scatter points, set alpha low to make points translucent

for g in np.unique(t):
    i = np.where(t == g)
    print(i, g)
    ax.scatter(embeddings2d[i,0], embeddings2d[i,1], label = ct[g],alpha = 0.7)
    ax.legend()
# ax.scatter(embeddingsdf.x, embeddingsdf.y, alpha=.5, c=t)
# ax.scatter(X[:,2], X[:,10], alpha=.5, c=t)
plt.title('t-SNE Scatter-Plot')

plt.show()
