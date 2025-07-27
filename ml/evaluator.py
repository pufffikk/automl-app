import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt


def explain_best_model(name, model, df, target_column):
    if name in ['RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM', 'CatBoost']:
        X = df.drop(columns=[target_column])
        X = pd.get_dummies(X)

        explainer = shap.Explainer(model)
        shap_values = explainer(X[:100])

        fig = plt.figure()
        shap.summary_plot(shap_values, X[:100], show=False)
        st.pyplot(fig)
    return f"Model {name} was selected because it achieved the best model evaluation metrics."
