from sklearn.manifold import Isomap

def embed(X, comp):
    embedding = Isomap(n_components=comp)
    X_transformed = embedding.fit_transform(X[:100])
    return X_transformed
