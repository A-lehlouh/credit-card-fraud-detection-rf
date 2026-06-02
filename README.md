Credit Card Fraud Detection — Random Forest

Pattern Recognition | Project 4 | Group 8
Al-Ahliyya Amman University · Course A0374502 · Dr. Ayman Mohamed


 Overview
This project implements a Random Forest classifier to detect fraudulent credit card transactions. The model is trained on a real-world, anonymized dataset of European cardholders and evaluated using standard classification metrics.
The dataset is highly imbalanced (fraudulent transactions are a tiny fraction of all records), making this a practical and challenging classification problem.

 Repository Structure
credit-card-fraud-rf/
│
├── pr4_CreditcardDataset.csv        # Dataset (31 features + Class label)
├── random_forest_credit_fraud.py    # Main training & evaluation script
├── confusion_matrix.png             # Generated after running the script
├── feature_importance.png           # Generated after running the script
├── Project_Report_Group8.docx       # Full written report
├── Presentation_Group8.pptx         # Project presentation slides
└── README.md

 Dataset
PropertyValueSourceAnonymized European credit card transactionsFeatures28 PCA components (V1–V28) + Time + AmountTargetClass — 0 = Normal, 1 = FraudTotal columns31

The feature names V1–V28 are anonymized via PCA to protect cardholder privacy.


 Methodology
Preprocessing

Drop missing values
Standardize all features using StandardScaler (mean = 0, std = 1)
Stratified train/test split: 80% train / 20% test

Model

Algorithm: Random Forest Classifier (sklearn)
Trees: 100 estimators
Max depth: 5 per tree
Random state: 42 (reproducible)

Evaluation

Accuracy score
Precision, Recall, F1-score (per class)
Confusion matrix
Top-10 feature importances


 Getting Started
Prerequisites
bashpip install pandas numpy scikit-learn matplotlib seaborn
Run
bashpython random_forest_credit_fraud.py
The script will print evaluation metrics to the console and save two output images:

confusion_matrix.png
feature_importance.png


 Results
After running the script you will see output similar to:
Accuracy: XX.XX%

Classification Report:
              precision    recall  f1-score   support
  Normal (0)       ...
   Fraud (1)       ...

Exact values depend on the dataset version used.


 Team — Group 8
Al-Ahliyya Amman University
Course: Pattern Recognition (A0374502)
Instructor: Dr. Ayman Mohamed

 License
This project is submitted for academic purposes only.
