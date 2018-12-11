import numpy as np
from sklearn.decomposition import PCA
"Calls the sklearn library for pca "
"and determines the intrinsic dimensionality"
def decompose(x,comp):
    pca = PCA(n_components=comp)
    pca.fit_transform(x)

    print(pca.explained_variance_ratio_)

    variance = pca.explained_variance_ratio_

    print(sum(variance)) #determine intrinsic dimensionality

    return pca.transform(x)