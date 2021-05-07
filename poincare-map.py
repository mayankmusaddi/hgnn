import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import random


colors_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
                  '#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#40004B',
                  '#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
                  '#D9F0D3', '#A6DBA0', '#5AAE61', '#1B7837', '#00441B',
                  '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3',
                  '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD',
                  '#CCEBC5', '#FFED6F', '#edf8b1', '#c7e9b4', '#7fcdbb',
                  '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']



cd = pd.read_csv('seqfish_plus/ge_withlabels.csv')
dt = {'Astrocyte' : 0, 'Choroid Plexus' : 1, 'Endothelial' : 2, 'Ependymal' : 3, 'Excitatory neuron' : 4, 'Interneuron' : 5, 'Microglia' : 6, 'Neural Stem' : 7, 'Neuroblast' : 8, 'Oligodendrocyte' : 9 }
ct = dict()
for i in dt:
  ct[dt[i]]=i
print(ct)
labels = (list(cd['labels']))
t = np.array([dt[x] for x in labels])
# t=list(t)

X = np.load('/home/anant/precog/hyp/Hyperbolic-GNNs/embeddings/embeddings.npy')
# X = np.zeros((913,4))

def plot_poincare_disc(x, labels=None, labels_name='labels', labels_order=None, 
                       file_name=None, coldict=None,
                       d1=10, d2=10, fs=11, ms=20, col_palette=plt.get_cmap("tab10").colors, bbox=(1.3, 0.7)):    
    # print(labels, type(labels), type(labels[0]))
    idx = np.random.permutation(len(x))
    df = pd.DataFrame(x[idx, :], columns=['pm1', 'pm2'])
    
    fig = plt.figure(figsize=(d1, d2))
    ax = plt.gca()
    circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)

    if not (labels is None):
        # print(labels[idx])
        df[labels_name] = labels[idx]
        if labels_order is None:
            labels_order = np.unique(labels)        
        if coldict is None:
            # print(list(col_palette.colors))
            coldict = dict(zip(labels_order, col_palette[:len(list(labels))]))
        sns.scatterplot(x="pm1", y="pm2", hue=labels_name, 
                        hue_order=labels_order,
                        palette=coldict,
                        alpha=1.0, edgecolor="none",
                        data=df, ax=ax, s=ms)

        ax.legend(fontsize=fs, loc='outside', bbox_to_anchor=bbox)
            
    else:
        sns.scatterplot(x="pm1", y="pm2",
                        data=df, ax=ax, s=ms)
    fig.tight_layout()
    ax.axis('off')
    ax.axis('equal')  

    labels_list = np.unique(labels)
    for l in labels_list:
#         i = np.random.choice(np.where(labels == l)[0])
        ix_l = np.where(labels == l)[0]
        c1 = np.median(x[ix_l, 0])
        c2 = np.median(x[ix_l, 1])
        ax.text(c1, c2, l, fontsize=fs)


    
    plt.savefig('I_am_trying.pdf', format='pdf')

    plt.close(fig)
plot_poincare_disc(X, labels=t)
