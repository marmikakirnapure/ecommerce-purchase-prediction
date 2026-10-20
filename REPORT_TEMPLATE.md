# E-Commerce Purchase Prediction Using Machine Learning

**Student name:** [Enter name]  
**Roll number:** [Enter roll number]  
**Course / class:** [Enter course]  
**Faculty:** [Enter faculty name]  
**GitHub repository:** [Paste your real repository URL]  
**Deployed website:** [Paste your real Streamlit URL]

## 1. Introduction and objectives
The project predicts whether an online shopping session results in a purchase (`Revenue`). The task is binary classification using the Online Shoppers Purchasing Intention Dataset. Objectives are to clean and explore the data, compare four classification models, evaluate the best model on held-out data, and provide a web interface for predictions.

## 2. Dataset
Dataset: Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository. The target is `Revenue`. The input includes browsing activity, timing, visitor and technical/traffic information. `PageValues` is excluded from predictors because it may be closely associated with completed transactions and could create target leakage.

Citation: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

## 3. Data cleaning and splitting
Describe the actual missing-value and duplicate counts from `outputs/cleaning_summary.json`. Explain duplicate removal, target conversion to 0/1, numerical median imputation and scaling, categorical mode imputation and one-hot encoding. State that the split is stratified: 80% training and 20% testing, with random state 42. Insert a screenshot of the cleaning summary and class distribution.

## 4. Models and methodology
The four models are Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree and Random Forest. Hyperparameters are selected using grid search with three-fold stratified cross-validation on training data. Preprocessing is kept inside each model pipeline to avoid fitting preprocessing on the test set.

Workflow: Dataset → cleaning and EDA → stratified train/test split → preprocessing → model tuning with cross-validation → test-set evaluation → Streamlit web app.

## 5. Results and performance analytics
Copy the actual rows from `outputs/model_comparison.csv` into the report. Do not invent or manually alter results.

| Model | CV F1 | Accuracy | Purchase Precision | Purchase Recall | Purchase F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Copy actual results] | | | | | | | |

Insert `outputs/model_comparison.png`, `outputs/confusion_matrix.png`, `outputs/roc_curve.png`, and `outputs/class_distribution.png`. Explain why purchase-class precision, recall, F1 and PR-AUC matter because the dataset is imbalanced. Identify the selected model from `outputs/experiment_summary.json`.

## 6. UI deployment and accessibility
Describe the Streamlit tabs for purchase prediction, model comparison and dataset information. Insert screenshots of the app form and prediction result. Record the actual GitHub repository and deployed Streamlit URL. Explain what you checked on desktop and mobile, and include screenshots from both. Only claim deployment/testing after completing it.

## 7. Conclusion, limitations and future work
Summarize the actual test results and which model was selected. Limitations include possible changes in customer behaviour, class imbalance, the dataset's age and the fact that the demo's session inputs may not be available at the beginning of a live browsing session. Future work could test external data, calibration and monitoring after deployment.

## 8. References
1. Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
2. Scikit-learn documentation: https://scikit-learn.org/stable/
3. Streamlit documentation: https://docs.streamlit.io/
4. Add at least five directly relevant research papers after verifying the complete papers and their reported results. Do not invent citations or findings.
