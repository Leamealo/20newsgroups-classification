import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def evaluate_selective_classification(model, X_test, y_test, thresholds):
    probs = model.predict_proba(X_test)
    max_probs = np.max(probs, axis=1) 
    predictions = np.argmax(probs, axis=1)

    results = []
    for tau in thresholds:
        accepted_mask = max_probs >= tau 
        coverage = np.mean(accepted_mask) 

        if np.sum(accepted_mask) > 0:
            f1_accepted = f1_score(y_test[accepted_mask], predictions[accepted_mask], average='macro')
        else:
            f1_accepted = 0.0

        results.append({
            'Threshold (tau)': tau,
            'Coverage (%)': round(coverage * 100, 2),
            'Manual Review (%)': round((1 - coverage) * 100, 2),
            'Macro-F1 (Accepted)': round(f1_accepted, 4)
        })

    return pd.DataFrame(results)
