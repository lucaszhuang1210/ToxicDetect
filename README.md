# ToxicDetect - Advanced Toxic Comment Classifier

**[Final Project Report (PDF)](./Advancing_Toxic_Comment_Classification.pdf)**  
*Advancing Toxic Comment Classification: A Comparative Study of TF-IDF, SBERT, and Deep Learning Models*  

---

A **machine learning pipeline** for detecting and classifying toxic comments across multiple categories.

This project investigates multiple approaches to **multi-label toxic comment classification** using the [Kaggle Toxic Comment Challenge dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). We compare traditional machine learning and deep learning pipelines to balance **performance, interpretability, and computational efficiency** for better content moderation.  

---

## Modeling Approaches  

Explored methods include:  
1. **TF-IDF + Naive Bayes** – baseline for imbalanced text data.  
2. **TF-IDF + Logistic Regression** – tuned linear model for improved performance.  
3. **SBERT + Logistic Regression / XGBoost** – semantic embeddings for richer representations.  
4. **RoBERTa Fine-Tuning** – end-to-end deep learning with contextualized features.  

Our objective: **evaluate and compare traditional vs. deep learning approaches** to identify the optimal strategy for scalable, reliable toxic language detection.  

---

## Result Summary  
- **Naive Bayes + TF-IDF**: ROC-AUC ≈ 0.945 — strong baseline, but limited flexibility.  
- **Logistic Regression + TF-IDF**: ROC-AUC ≈ 0.976 — best traditional model, efficient and interpretable.  
- **SBERT + Logistic Regression**: ROC-AUC ≈ 0.969 — better semantic representation, but only modest gain.  
- **SBERT + XGBoost**: ROC-AUC ≈ 0.959 — leveraging non-linear boosting, but not top performer.  
- **RoBERTa Fine-Tuning**: ROC-AUC ≈ 0.978 — highest score, best contextual understanding, but computationally expensive.  

**Conclusion:** TF-IDF with Logistic Regression remains a highly effective baseline, but fine-tuned transformer models (e.g., RoBERTa) provide the best performance when resources allow.  

---

## 📂 Project Structure  
```
.
├── data
│   ├── sample_submission.csv
│   ├── test_labels.csv
│   ├── test.csv
│   └── train.csv
├── models
│   ├── generate_embeddings.py
│   ├── RoBERTa.py
│   ├── SBERT_LogisticRegression.ipynb
│   └── TFIDF_LogisticRegression_Optimized.ipynb
├── Advancing Toxic Comment Classification.pdf
└── README.md
```
