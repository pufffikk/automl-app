import streamlit as st
import pandas as pd
import tempfile
import dask.dataframe as dd
from sklearn.metrics import ConfusionMatrixDisplay

from ml.ml_functions import detect_task_type
from ml.tabular_models import train_tabular_models
from ml.nlp_model import train_nlp_model
from ml.evaluator import explain_best_model
from utils.model_saver import save_model_as_pickle
from utils import db_logger
import matplotlib.pyplot as plt

st.set_page_config(page_title="AutoML App", layout="wide")
st.title("AutoML Web App")

db_logger.init_db()

uploaded_file = st.file_uploader("Upload CSV or Excel file (max 100MB)", type=["csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    if uploaded_file.name.endswith('.csv'):
        df = dd.read_csv(file_path)
    else:
        df_pandas = pd.read_excel(file_path)
        df = dd.from_pandas(df_pandas, npartitions=4)

    st.subheader("Preview of Data")
    if hasattr(df, 'compute'):
        st.dataframe(df.compute().head())
    else:
        st.dataframe(df.head())

    if hasattr(df, 'compute'):
        df_pd = df.compute()
    else:
        df_pd = df

    task_type, text_column = detect_task_type(df_pd)
    st.info(f"Detected task type: {task_type.upper()}")

    target_column = st.selectbox("Select target column", df_pd.columns)

    if 'history' not in st.session_state:
        st.session_state.history = []

    if st.button("Train Models"):
        with st.spinner("Training in progress..."):
            if task_type == 'tabular':
                results, best_model_name, best_model, best_cm = train_tabular_models(df_pd, target_column)
            else:
                results, best_model_name, best_model, best_cm = train_nlp_model(df_pd, target_column, text_column)

        st.success(f"Best model: {best_model_name}")

        st.write("Evaluation Results:")
        st.dataframe(results)

        disp = ConfusionMatrixDisplay(confusion_matrix=best_cm)
        fig, ax = plt.subplots(figsize=(4, 3))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        st.pyplot(fig)

        acc = results.loc[results['Model'] == best_model_name, 'Accuracy'].values[0]
        roc = results.loc[results['Model'] == best_model_name, 'ROC AUC'].values[0]

        db_logger.log_run(best_model_name, acc, roc)

        st.session_state.history.append({
            'model': best_model_name,
            'accuracy': acc,
            'roc_auc': roc,
        })

        st.subheader("Run History (this session)")
        st.table(st.session_state.history)

        explanation = explain_best_model(best_model_name, best_model, df_pd, target_column)
        st.write("Explanation:", explanation)

        pkl_bytes = save_model_as_pickle(best_model)
        st.download_button("Download Best Model", pkl_bytes, file_name="best_model.pkl")

    st.subheader("All Runs History (from SQLite)")
    all_runs = db_logger.get_all_runs()
    st.write(all_runs)
