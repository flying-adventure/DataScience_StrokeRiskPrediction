from sklearn.linear_model import LogisticRegression

def get_model():
    """
    Returns a Logistic Regression model with:
    - C = 1 (Regularization strength)
    - penalty = 'l2'
    - solver = 'liblinear'
    """
    return LogisticRegression(
        C=1,
        penalty='l2',
        solver='liblinear',
        random_state=42
    )
