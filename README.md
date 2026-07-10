# 🤖 Machine Learning Projects

A collection of machine learning projects spanning classification, natural language processing, computer vision, anomaly detection, and cybersecurity. Each project covers the full pipeline from data preprocessing and feature engineering through model training and evaluation, with a focus on real-world problem framing and clean, reproducible code.

---

## 📂 Projects

| # | Project | Domain | Key Methods |
|---|---------|--------|-------------|
| 1 | [💳 Credit Card Fraud Detection](#1-credit-card-fraud-detection) | Finance | Logistic Regression, Random Forest, class imbalance handling |
| 2 | [📧 Spam Email Classifier](#2-spam-email-classifier) | NLP | TF-IDF, Naive Bayes, SVM |
| 3 | [🛡️ DDoS Attack Classification](#3-ddos-attack-classification) | Network Security | Random Forest, Neural Networks, ROC-AUC |
| 4 | [🔗 URL Phishing Detection](#4-url-phishing-detection) | Cybersecurity | TF-IDF, Logistic Regression, custom tokenizer |

---

## 1. 💳 Credit Card Fraud Detection

**[View Project](./Project_1_Credit_card_Fraud_Detection_with_Machine_learning_Project/)**

This project tackles the detection of fraudulent credit card transactions using supervised machine learning. The dataset is severely class-imbalanced (fraud is rare by nature), so a significant part of the work goes into handling that imbalance correctly before any model is trained. Multiple classifiers are compared, including Logistic Regression, Decision Trees, and Random Forest, and performance is measured using F1 Score and Precision-Recall curves rather than raw accuracy, which would be misleading on imbalanced data.

**Highlights:**
- Imbalance handling via resampling and class weighting
- Precision-Recall and F1 evaluation for imbalanced classification
- Comparison across multiple classifiers

---

## 2. 📧 Spam Email Classifier

**[View Project](./Project_2_Spam_Email_Classifier_using_Machine_Learning/)**

A text classification pipeline that distinguishes spam from legitimate email using natural language processing. Raw email text is cleaned and transformed into numerical features using TF-IDF and N-gram representations, then fed into Naive Bayes and SVM classifiers. The project demonstrates how NLP preprocessing choices (vocabulary size, n-gram range, stop-word removal) interact with model performance, and evaluates results using accuracy, F1 score, and a confusion matrix.

**Highlights:**
- TF-IDF vectorisation with N-gram feature extraction
- Naive Bayes and SVM classifiers compared
- Full evaluation, including confusion matrix breakdown

---

## 3. 🛡️ DDoS Attack Classification

**[View Project](./Project_3_DDoS_Attack_Classification_using_Machine_Learning/)**

Network intrusion detection framed as a multi-class classification problem using the CICIDS 2017 dataset, one of the standard benchmarks for network security ML research. The project covers exploratory data analysis on high-dimensional network traffic features, preprocessing to handle noise and class imbalance across attack types, and training Random Forest, Logistic Regression, and Neural Network models. Evaluation uses accuracy, F1 score, recall, and ROC-AUC to give a complete picture of detection reliability.

**Highlights:**
- CICIDS 2017 dataset (real-world network traffic captures)
- Multi-class attack type classification
- Neural network baseline alongside classical ML models
- ROC-AUC evaluation for reliable multi-class assessment

---

## 4. 🔗 URL Phishing Detection

**[View Project](./Project%204%20-%20Implement%20URL%20phishing%20detection%20with%20Logistic%20Regression%20and%20TF-IDF%20vectorization/)**

A URL classification system that identifies phishing sites by treating URLs as text and applying TF-IDF vectorisation with a custom tokeniser that strips protocol prefixes and non-alphanumeric characters before feature extraction. A Logistic Regression model is trained on the resulting feature vectors to classify URLs as legitimate or phishing. The trained model and vectorizer are serialised with pickle so new URLs can be scored at inference time without retraining. The project demonstrates how NLP feature extraction techniques transfer cleanly to non-text domains like URL analysis.

**Highlights:**
- Custom URL tokeniser for cleaner feature extraction
- TF-IDF applied to URLs as character/token sequences
- Pickle serialisation for production-ready inference
- No retraining needed for new URL classification

---

## 🧰 Tech Stack

`Python` `scikit-learn` `pandas` `NumPy` `Matplotlib` `Seaborn` `NLTK` `Jupyter Notebook`

---

## 📄 License

MIT
