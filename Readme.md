# Interactive KNN Classifier – Streamlit App

This project is an **interactive K-Nearest Neighbors (KNN) classifier** built using **Streamlit**.  
It allows users to **experiment with different K values** and observe how the model's performance metrics — **accuracy**, **confusion matrix**, and **ROC–AUC curve** — change dynamically.

---

## Overview

This app helps visualize how **KNN hyperparameter tuning** affects model performance.  
For this dataset, **K = 28** gives the best overall accuracy and generalization.

---

## Features

- **Interactive Slider** – Adjust the value of *K* (1 to 50) in real-time.  
- **Accuracy & Confusion Matrix** – Updated automatically with each K change.  
- **ROC–AUC Curve Visualization** – View how classification quality evolves with different K.  
- **Automatic Model Save** – Model trained at *K=28* is saved as `knn_model.pkl`.  
- **Clean UI** – Built fully in Streamlit with no dataset upload required.

---


- **Python 3.11+**  
- **Streamlit** – Interactive web UI  
- **scikit-learn** – KNN modeling and evaluation  
- **matplotlib / seaborn** – Visualization  
- **pandas / numpy** – Data handling and processing  
