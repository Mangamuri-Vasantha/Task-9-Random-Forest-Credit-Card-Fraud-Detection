# Task-9-Random-Forest-Credit-Card-Fraud-Detection
# 💳 Credit Card Fraud Detection using Random Forest

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using **Machine Learning** techniques.  
Since fraud datasets are highly **imbalanced**, traditional accuracy is misleading.  
We use **precision, recall, and F1-score** as evaluation metrics and apply **SMOTE** to handle class imbalance.

---

## 🧰 Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

---

## 📂 Dataset
- Synthetic Credit Card Fraud Dataset  
- Target column: `Class`
  - `0` → Non-Fraud
  - `1` → Fraud  

The dataset is highly imbalanced, with very few fraud cases compared to non-fraud transactions.

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading & Exploration
- Loaded dataset using Pandas
- Checked class distribution to understand imbalance

### 2️⃣ Data Preprocessing
- Separated features and target variable
- Performed train-test split using **stratified sampling**
- Applied feature scaling using `StandardScaler`

### 3️⃣ Baseline Model – Logistic Regression
- Trained Logistic Regression as a baseline model
- Observed high accuracy but poor fraud detection
- Learned why accuracy is misleading for imbalanced datasets

### 4️⃣ Random Forest Model
- Trained Random Forest classifier
- Used ensemble learning to improve performance
- Initially failed to detect fraud due to imbalance

### 5️⃣ Handling Imbalanced Data (SMOTE)
- Applied **SMOTE** only on training data
- Balanced fraud and non-fraud samples
- Retrained Random Forest on balanced data

### 6️⃣ Model Evaluation
- Evaluated models using:
  - Precision
  - Recall
  - F1-score
- Focused on improving **fraud recall**

### 7️⃣ Feature Importance
- Plotted feature importance from Random Forest
- Identified key features contributing to fraud detection

### 8️⃣ Model Saving
- Saved trained Random Forest model using `joblib`
- Saved scaler for future inference

---
## ✅ Conclusion
This project demonstrates the importance of handling class imbalance in real-world ML problems like fraud detection.  
Using ensemble models and resampling techniques leads to more reliable and practical results.

