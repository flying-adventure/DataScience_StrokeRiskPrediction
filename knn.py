from sklearn.neighbors import KNeighborsClassifier

def get_model():
    return KNeighborsClassifier(
        n_neighbors=13,
        weights='uniform',
        p=2
    )
