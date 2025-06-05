from sklearn.linear_model import LogisticRegression

def get_model():
    """
    Returns a Logistic Regression model with:
    - C = 0.1 (stronger regularization)
    - penalty = 'l1'
    - solver = 'liblinear'
    """
    return LogisticRegression(
        C=0.1,
        penalty='l1',
        solver='liblinear',
        random_state=42
    )
