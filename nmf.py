from sklearn.decomposition import NMF

"Calls the sklearn library for non-negative matrix decomposition"

def decompose(X, comp):
    nmf = NMF(n_components=comp)

    X_transformed = nmf.fit_transform(X)

    return X_transformed
