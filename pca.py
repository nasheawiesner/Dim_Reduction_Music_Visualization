import numpy as np
from sklearn.decomposition import PCA

def decompose(x):
    pca = PCA(n_components=7)
    pca.fit(x)

    print(pca.explained_variance_ratio_)

    variance = pca.explained_variance_ratio_

    print(sum(variance)) #determine intrinsic dimensionality

    print(pca.transform(x))

    return pca.transform(x)