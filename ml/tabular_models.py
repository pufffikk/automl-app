from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
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

from ml.ml_functions import confusion_matrix_to_array


def train_tabular_models(df, target_column):
    df = df.dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X)
    if y.dtype.kind not in 'biufc':
        y = LabelEncoder().fit_transform(y)

    print(f"TEST {y}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(),
        'CatBoost': cb.CatBoostClassifier(verbose=0),
        'KNN': KNeighborsClassifier(),
        'MLP': MLPClassifier(max_iter=500),
        'NaiveBayes': GaussianNB()
    }

    if len(X_train) < 10000:
        models["SVM"] = SVC(probability=True)

    results = []
    best_score = 0
    best_model_name = ''
    best_model = None
    best_cm = None

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(set(y)) == 2 else None
            cm = confusion_matrix(y_test, y_pred)
            cm_flattered = confusion_matrix_to_array(cm)
            results.append({'Model': name, 'Accuracy': acc, 'ROC AUC': auc, 'Confusion Matrix': cm_flattered})
            if acc > best_score:
                best_score = acc
                best_model_name = name
                best_model = model
                best_cm = cm
        except Exception as e:
            results.append({'Model': name, 'Accuracy': None, 'ROC AUC': None})
            traceback.print_exc()
            continue

        print(f"Training {name}... finished")

    return pd.DataFrame(results), best_model_name, best_model, best_cm
