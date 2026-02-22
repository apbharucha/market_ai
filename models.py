import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(X, y):
    n_classes = int(len(np.unique(y)))
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        objective='multi:softprob' if n_classes >= 3 else 'binary:logistic',
        num_class=n_classes if n_classes >= 3 else None,
        eval_metric='mlogloss',
        random_state=42
    )

    rf.fit(X, y)
    xgb.fit(X, y)

    return rf, xgb
