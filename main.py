#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt





feature_labels = ['animal name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type']
df = pd.read_csv('./zoo.data', sep=',', names = feature_labels)


# In[3]:


df = df.drop(columns=['animal name'])
df


# In[4]:


df = pd.concat([df,pd.get_dummies(df['legs'], prefix='legs')], axis=1)
df = df.drop(columns=("legs"))


# In[5]:


df_labels = df["type"]
df = df.drop(columns=['type'])


# In[6]:


X = df.to_numpy()
X.shape


# In[50]:


def plot_reduced(reduced, labels, model):
    plt.figure()
    for i in range(1, 8):
        plt.scatter(reduced[labels == i, 0], reduced[labels == i, 1], label=i)

    plt.legend(loc='upper left')
    plt.title(model)
    
    plt.show()


# In[51]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)


    
plot_reduced(X_reduced, df_labels, 'PCA')


# In[52]:


import numpy as np



def get_S(D):
    one_mat = np.ones((D.shape[0],1))
    print(one_mat.shape)
    E1 = np.dot(np.dot(D, one_mat), one_mat.T)/D.shape[0]
    colm = np.dot(one_mat, np.dot(one_mat.T, D))
    E2 = colm / D.shape[0]
    E3 = np.dot(np.dot(colm, one_mat), one_mat.T) / D.shape[0] ** 2
    S = -(D - E1 - E2 + E3)/2
    return S


def mds  (X, importance_factor, n_components=2):





    D = []
    for i in range(X.shape[0]):
        D.append([])
        for j in range(X.shape[0]):
            d = sqrt(np.sum(np.multiply(importance_factor[:len(X[i, :])],np.square(X[i, :] -  X[j, :]) ) ))
            D[i].append(d)
    D =  np.array(D)
    S = get_S(D)
    eigenvalues,eigenvectors = np.linalg.eig(S)
    Diag = np.diag(eigenvalues)
    return eigenvectors[:,:2].dot(Diag[:2,:2])

print("plotting")
plot_reduced(mds(X, importance_factor), df_labels, 'MDS')


# In[62]:


def get_h( X):
    D = cdist(X,X,'euclidean')
    h = np.ones((D.shape[0], D.shape[0])) * float('inf')
    for i in range(D.shape[0]):
        k = np.argsort(D[i, :])[:n_neighbors + 1]
        h[i][k] = D[i][k]
    return h



# In[61]:


from scipy.spatial.distance import cdist
from utils import dijkstra

importance_factor = [1]*101



def isomap( X, n_neighbors = 10):
    D = cdist(X,X,'euclidean')
    h = get_h(D)
    
    for i in range(D.shape[0]):
        h = dijkstra(h, i, n_neighbors)
        
        
    to_inf = np.where(  np.isinf(h))
    h[to_inf] = D[to_inf]
    return mds(h, importance_factor)
    
i = 10
plot_reduced(isomap(X), df_labels, ' Isomap ' +str(i))


# In[63]:


i = 2
plot_reduced(isomap(X, i), df_labels, ' Isomap ' +str(i))


for i in range(10,100,10):
    plot_reduced(isomap(X, i), df_labels, ' Isomap ' +str(i))


# In[ ]:




