import numpy as np
from sklearn.decomposition import PCA

def decompose(x,comp):
    pca = PCA(n_components=comp)
    pca.fit_transform(x)

    #print(pca.explained_variance_ratio_)

    #variance = pca.explained_variance_ratio_

    #print(sum(variance)) #determine intrinsic dimensionality

    #print(pca.transform(x))

    return pca.transform(x)