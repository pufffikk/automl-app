import pandas as pd
from sklearn.datasets import make_classification
from ml.tabular_models import train_tabular_models


def test_train_tabular_models():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
    df['target'] = y
    results, best_model_name, best_model, best_cm = train_tabular_models(df, 'target')
    assert not results.empty
    assert best_model_name in results['Model'].values
