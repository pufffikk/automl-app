# AutoML Web Application

This project is a lightweight AutoML web application that allows users to:

- Upload tabular or text classification datasets (`.csv` or `.xlsx`)
- Automatically detect task type (tabular or NLP)
- Train multiple machine learning models
- Evaluate and explain the results
- Download the best model as a `.pkl` file

---

## 🚀 Features

✅ Upload datasets up to 100MB  
✅ Auto-detect classification type: **Tabular** or **Text (NLP)**  
✅ Train models:
- **Tabular:** Logistic Regression, Random Forest, ExtraTrees, XGBoost, LightGBM, CatBoost, SVM, KNN, MLP, Naive Bayes
- **NLP:** DistilBERT (via Hugging Face Transformers)  
✅ Evaluate using **Accuracy**, **ROC AUC**, **Confusion Matrix**  
✅ **Explain** tree-based models using **SHAP**  
✅ Download best model as `.pkl`  
✅ Store run history in SQLite  
✅ Session history with Streamlit state  
✅ Handle larger CSV datasets with Dask  
✅ Hosted online for free with no API keys required

---

## 📦 Installation (for local use)

```bash
git clone https://github.com/yourusername/automl-app.git
cd automl-app
pip install -r requirements.txt
streamlit run app.py
