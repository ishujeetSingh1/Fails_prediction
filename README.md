# Fails_prediction

## Project Objective: 
1. Developing a reliable classification model to distinguish between “pass” and “fail” cases in a highly imbalanced dataset, where “fail” is the minority class.
2. Reducing false negatives (i.e., missed fails) is a critical priority—ensuring that true fails are caught whenever possible—while still keeping overall performance (e.g., balanced accuracy, precision) at an acceptable level.
3. Implementing and evaluating different techniques—like XGBoost, oversampling (SMOTE), PCA, and hyperparameter tuning—to find the best balance between catching fails and avoiding excessive false positives.

## Key Finding 
### Imbalanced Data Challenge
The dataset is highly imbalanced, with around 7% of samples in the "fail" class.
This imbalance made it critical to employ oversampling techniques (SMOTE) and class balancing strategies (e.g., scale_pos_weight).

### Pipeline Components and Tuning
A combination of scaling, oversampling, dimensionality reduction (PCA), and a robust classifier like XGBoost provided the best results.
Hyperparameter tuning with RandomizedSearchCV was crucial in finding the optimal settings for the pipeline.

### Threshold Adjustment
By adjusting the decision threshold, we could prioritize reducing false negatives (missed "fails") at the cost of increasing false positives.

### XGBoost with Oversampling.
XGBoost, when paired with SMOTE and class weighting (scale_pos_weight), consistently outperformed simpler classifiers.
This method balanced the trade-off between minority-class recall (fail detection) and overall performance.

## Best score & Interpretation
### Best Balanced Accuracy
Cross-Validation (CV): 0.7601
Test Set: 0.8435 (post-threshold tuning)
### Confusion Matrix
[[285   8]
 [  6  15]]
True Negatives (285): Correctly predicted "pass" samples.
False Positives (8): "Pass" samples incorrectly predicted as "fail."
False Negatives (6): "Fail" samples incorrectly predicted as "pass" (missed fails).
True Positives (15): Correctly predicted "fail" samples.
The low false negatives (6) demonstrate the model's ability to detect fails effectively, a critical goal of the project.


DataSet from https://archive.ics.uci.edu/dataset/179/secom
