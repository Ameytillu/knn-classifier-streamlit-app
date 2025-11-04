import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    roc_curve, roc_auc_score
)

st.markdown("""
### 📘 About this App  
This interactive KNN Classification App is designed to visualize how different **K values** affect model performance on the given dataset.  
You can use the **slider in the sidebar** to adjust the value of *K* and observe how the **accuracy**, **confusion matrix**, and **ROC–AUC curve** change.  

For this dataset, **K = 28** provides the best balance between accuracy and generalization.
""")


# -----------------------------------------------------------
# 🎯 Page Setup
# -----------------------------------------------------------
st.set_page_config(page_title="KNN Classifier App", layout="centered")
st.title("🔍 Interactive KNN Classifier")

# -----------------------------------------------------------
# 📘 Load Dataset
# -----------------------------------------------------------
@st.cache_data
def load_data():
    # Try a few common filenames (the dataset in this repo is named without .csv)
    candidates = [
        "knn_assignment_data.csv",
        "knn_assignment_data",
        "knn_assignment_data.txt",
    ]
    for fname in candidates:
        try:
            df = pd.read_csv(fname)
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            # Surface parsing errors to the Streamlit app
            st.error(f"Error reading {fname}: {e}")
            raise

    st.error("Dataset file not found. Place `knn_assignment_data` or `knn_assignment_data.csv` in the app folder.")
    st.stop()

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------------------------------------
# ⚙️ Prepare Data
# -----------------------------------------------------------
X = df.drop('TARGET CLASS', axis=1)
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------
# 🎚️ Sidebar Slider for K Value
# -----------------------------------------------------------
st.sidebar.header("Model Configuration")
k = st.sidebar.slider("Select K Value", min_value=1, max_value=50, value=5)
st.sidebar.write(f"Current K = {k}")

# -----------------------------------------------------------
# 🤖 Train Model
# -----------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
y_prob = knn.predict_proba(X_test_scaled)[:, 1]

# -----------------------------------------------------------
# 📈 Evaluation Metrics
# -----------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {acc:.3f}")
st.write(f"**AUC Score:** {auc:.3f}")

# Display Confusion Matrix
st.write("**Confusion Matrix:**")
st.dataframe(pd.DataFrame(cm,
                          columns=["Predicted 0", "Predicted 1"],
                          index=["Actual 0", "Actual 1"]))

# -----------------------------------------------------------
# 📉 ROC Curve
# -----------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label=f'KNN (AUC = {auc:.3f})')
ax.plot([0,1], [0,1], color='red', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curve (K={k})')
ax.legend()
st.pyplot(fig)


