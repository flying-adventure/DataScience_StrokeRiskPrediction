from sklearn.ensemble import RandomForestClassifier

def get_model():
    """
    Returns a Random Forest model with:
    - n_estimators = 200
    - max_depth = 20 (limited depth)
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )
