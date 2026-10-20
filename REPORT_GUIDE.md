# 10-Page Report Planning Guide

Use your professor's required headings and adapt the page allocation to your institution's format.

## Page 1 — Introduction
- E-commerce purchase intention problem
- Motivation and practical relevance
- Objectives and scope
- Research questions

## Pages 2–3 — Literature Review
Discuss at least five recent, directly relevant research papers. For every paper record:
- Full citation, year and publication venue
- Dataset and sample size
- Algorithms compared
- Metrics used and reported findings
- Limitations and how the current project differs

Compare the papers in a table. Verify claims against the full paper; do not invent results.

## Page 4 — Methodology
Insert a workflow diagram:
Dataset → data validation and cleaning → stratified 80:20 split → preprocessing pipeline → 3-fold CV/grid search on training data → four models → held-out test evaluation → Streamlit UI.

## Page 5 — Data Cleaning and Preprocessing
Use `outputs/cleaning_summary.json` for the actual counts. Describe:
- Missing values and duplicates observed
- Target distribution and class imbalance
- `Revenue` target encoding
- Exclusion of `PageValues` due to leakage risk
- Numeric scaling and categorical one-hot encoding
- Why preprocessing is inside the pipeline

## Page 6 — Models and Experimental Setup
Explain Logistic Regression, KNN, Decision Tree and Random Forest. Include the hyperparameter grids from `train.py`, random state 42, 80:20 stratified split, and three-fold stratified CV.

## Pages 7–8 — Results and Discussion
Use actual values from `outputs/model_comparison.csv`. Include:
- Model metrics table
- `model_comparison.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `class_distribution.png`
- Discussion of precision, recall, F1, ROC-AUC and PR-AUC
- Error analysis and limitations

Do not report test-set results as training performance. The selected model is selected using training cross-validation F1, not test scores.

## Page 9 — UI Development and Deployment
Include screenshots of the prediction form and results tab, explain how the model is loaded, and add your working public URL. Test the same URL on a laptop and phone browser.

## Page 10 — Conclusion and References
Summarize the observed result, limitations and realistic future work. List dataset citation, five research papers, and any software/documentation references in one consistent citation style.

## Official dataset citation
Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
