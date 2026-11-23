# ML Lab 4 - Ticket Escalation Classification

Binary classification of synthetic support tickets to predict whether a ticket gets escalated (`escalated` = 1).

## Files
- `Lab_4.ipynb` - notebook with code, outputs, answers and recommendation
- `classification_escalation.csv` - dataset (900 tickets)

## What the notebook covers
- Data inspection and a stratified 75/25 train/test split
- Preprocessing: median / most-frequent imputation, scaling, one-hot encoding
- Models: majority baseline, logistic regression, LDA, k-NN
- Confusion matrix, ROC curve, and threshold table with cost = 5*FN + FP
- Stratified 5-fold CV to tune k for k-NN

## Result
Logistic regression with a 0.20 threshold had the lowest error cost (98, vs 188 at the default 0.5).

## How to run
    pip install numpy pandas matplotlib scikit-learn
Open `Lab_4.ipynb` in VS Code or Jupyter and click Run All. Keep the CSV in the same folder as the notebook.
