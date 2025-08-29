---
applyTo: '**'
---

Project Title: Student Performance Classification using Machine Learning

Problem Statement:
Pearson VUE, a global leader in computer-based testing, aims to classify students into categories such as High Performer, Average Performer, and Needs Improvement. Using a dataset containing Mid-Semester Test (MST) scores, Quiz results, Attendance records, and Assignment performance, you are required to develop a Python-based machine learning classification model. The project should include preprocessing of the dataset such as handling missing values and normalizing features, training and evaluating multiple classification algorithms, and comparing their performance to select the best model for accurate student categorization.

Instructions Regarding Project:
1. Dataset:
   - Dataset can be self-generated or obtained from standard platforms (e.g., Kaggle, GitHub).
   - Number of features can be selected as per project scope and relevance.
   - Example Features:
        * MST_Score
        * Quiz_Avg
        * Attendance (%)
        * Assignment_Score
   - Target Variable:
        * Category (High Performer, Average Performer, Needs Improvement)

2. Preprocessing:
   - Handle missing values (e.g., imputation).
   - Normalize/standardize numeric features.
   - Encode categorical target labels into numerical classes.

3. Exploratory Data Analysis (EDA):
   - Visualize score distributions (histograms, boxplots).
   - Correlation heatmap to study relationships between features.
   - Attendance vs. Performance trend analysis.

4. Model Building:
   - Perform train-test split (e.g., 70-30 or 80-20).
   - Train multiple ML models such as:
        * Logistic Regression
        * Decision Tree
        * Random Forest
        * Support Vector Machine (SVM)
        * K-Nearest Neighbors (KNN)
        * Gradient Boosting / XGBoost (optional)
   - Hyperparameter tuning can be included for better performance.

5. Model Evaluation:
   - Metrics to be considered:
        * Accuracy
        * Precision, Recall, F1-Score
        * Confusion Matrix
        * ROC-AUC curve (for binary/multiclass cases)
   - Perform k-fold cross validation if possible.

6. Model Comparison:
   - Compare performance of models using evaluation metrics.
   - Select the best performing model for final deployment.

7. Visualization of Results:
   - Graphs for model comparison (bar chart of accuracy/F1-score).
   - Confusion matrix heatmaps.
   - ROC curves.

8. Project Deliverables:
   - Python script or Jupyter Notebook implementing full functionality.
   - Dataset (CSV or similar format).
   - Documentation with:
        * Problem Statement
        * Data preprocessing steps
        * EDA visualizations
        * Model training and evaluation results
        * Flow diagram of the process
   - Prediction/Inference script:
     * A small CLI script (example: `src/predict.py`) that loads the saved model artifacts and either predicts a CSV of rows or a single sample and writes predictions to an output CSV.
     * Usage example should be provided in the README (e.g., `python src/predict.py --input data/students.csv --output outputs/predictions.csv` or single-sample flags for ad-hoc predictions).

9. Suggested Flow Diagram (Text Representation):
   Start
     │
     ▼
   Data Collection → Dataset (MST, Quiz, Attendance, Assignment)
     │
     ▼
   Data Preprocessing
       - Handle Missing Values
       - Normalize/Standardize Features
       - Encode Labels
     │
     ▼
   Exploratory Data Analysis (EDA)
       - Visualization
       - Correlation Analysis
     │
     ▼
   Model Building
       - Train/Test Split
       - Apply ML Models
     │
     ▼
   Model Evaluation
       - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
     │
     ▼
   Best Model Selection
     │
     ▼
   Deployment / Report
     │
     ▼
   End

This file contains all instructions required to implement the Student Performance Classification project using Machine Learning in Python.
