from sklearn.decomposition import NMF


def decompose(X, comp):
    nmf = NMF(n_components=comp)

    X_transformed = nmf.fit_transform(X)

    return X_transformed
