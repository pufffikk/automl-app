from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pandas as pd
import traceback


def train_tabular_models(df, target_column):
    df = df.dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X)
    if y.dtype.kind not in 'biufc':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier(),
        'LightGBM': lgb.LGBMClassifier(),
        'CatBoost': cb.CatBoostClassifier(verbose=0),
        'LogisticRegression': LogisticRegression(max_iter=500),
        'KNN': KNeighborsClassifier(),
        'MLP': MLPClassifier(max_iter=500),
        'NaiveBayes': GaussianNB()
    }

    if len(X_train) < 10000:
        models["SVM"] = SVC(probability=True)

    results = []
    best_model = None
    best_model_name = None

    def is_better(m1, m2):
        # Priority: recall > precision > f1 > accuracy
        order = ['recall', 'precision', 'f1', 'accuracy']
        for key in order:
            v1 = m1.get(key)
            v2 = m2.get(key)

            if v1 is None:
                return False
            if v2 is None:
                return True
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        return False

    best_metrics = {'recall': -1, 'precision': -1, 'f1': -1, 'accuracy': -1}

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(set(y)) == 2 else None
            cm = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results.append({
                'Model': name,
                'Accuracy': acc,
                'ROC AUC': auc,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Confusion Matrix': cm.tolist()
            })

            current_metrics = {'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': acc,
                               'confusion_matrix': cm}
            if is_better(current_metrics, best_metrics):
                best_metrics = current_metrics
                best_model_name = name
                best_model = model

        except Exception:
            results.append({'Model': name, 'Accuracy': None, 'ROC AUC': None, 'Precision': None, 'Recall': None, 'F1': None})
            traceback.print_exc()
            continue

        print(f"Training {name}... finished")

    return pd.DataFrame(results), best_model_name, best_model, best_metrics.get('confusion_matrix')

