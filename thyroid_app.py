#thyroid app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Thyroid Cancer Predictor", layout="wide")

# --- TITLE ---
st.markdown("<h1 style='text-align: center; color: purple;'>🔬 Thyroid Cancer Recurrence Predictor</h1>", unsafe_allow_html=True)

# --- LOAD DATA ---
df = pd.read_csv(r"C:\Users\GS Suraj Rao\Downloads\Thyroid_cancer\Thyroid_Diff.csv")

# --- ENCODING ---
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# --- SPLIT FEATURES ---
X = df.drop("Recurred", axis=1)
y = df["Recurred"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL TRAINING ---
with st.spinner("Training model..."):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "📈 Model Insights", "🔮 Prediction"])

# ===========================
# 📊 TAB 1: DATA EXPLORATION
# ===========================
with tab1:
    st.subheader("Preview Dataset")
    if st.checkbox("Show full dataset"):
        st.dataframe(df)
    else:
        st.dataframe(df.head())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ========================
# 📈 TAB 2: MODEL INSIGHTS
# ========================
with tab2:
    st.subheader("Model Performance")
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="🔍 Accuracy", value=f"{acc:.2f}")

    st.text("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders['Recurred'].classes_)
    disp.plot(ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Importance")
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.plot(kind='barh', ax=ax, color="skyblue")
    st.pyplot(fig)

# ========================
# 🔮 TAB 3: PREDICTION TOOL
# ========================
with tab3:
    st.subheader("Input Patient Details for Prediction")
    user_input = {}

    for col in X.columns:
        if col == "Age":
            user_input[col] = st.slider("Age", int(df[col].min()), int(df[col].max()), int(df[col].mean()))
        else:
            options = label_encoders[col].classes_.tolist()
            user_input[col] = st.selectbox(f"{col}", options)

    # Encode input
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if col != "Age":
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    if st.button("🔍 Predict Recurrence"):
        with st.spinner("Predicting..."):
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            result = label_encoders['Recurred'].inverse_transform([prediction])[0]
            confidence = np.max(proba) * 100

            st.success(f"Predicted Recurrence: {result}")
            st.info(f"Confidence: {confidence:.2f}%")

# Optional: Export report
st.sidebar.header("📥 Download Reports")
csv = report_df.to_csv(index=True).encode("utf-8")
st.sidebar.download_button("Download Classification Report", csv, "report.csv", "text/csv")
