from sklearn.ensemble import RandomForestClassifier

def get_model():
    """
    Returns a Random Forest model with:
    - n_estimators = 100
    - max_depth = None (unlimited depth)
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
