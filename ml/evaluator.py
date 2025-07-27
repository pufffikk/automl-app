import pandas as pd
import shap
import streamlit as st


def explain_best_model(name, model, df, target_column):
    if name in ['RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM', 'CatBoost']:
        X = df.drop(columns=[target_column])
        X = pd.get_dummies(X)
        explainer = shap.Explainer(model)
        shap_values = explainer(X[:100])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, X[:100])
        st.pyplot()
    return f"Model {name} was selected because it achieved the best accuracy."
