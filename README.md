Breast Cancer Classification Using Machine Learning

Problem Statement:

The goal of this assignment is to develop and compare multiple machine learning classification models to predict whether a tumor is malignant or benign based on the Breast Cancer Wisconsin (Diagnostic) dataset. By analyzing various models and their performance, we can understand which algorithm works best for this dataset and make predictions with reasonable accuracy.

Dataset Description:

The dataset used in this assignment is the Breast Cancer Wisconsin (Diagnostic) dataset, available from UCI Machine Learning Repository. It contains 569 instances and 32 features including measurements like radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

Target column: diagnosis (M = malignant, B = benign)

Input features: 30 numeric features describing tumor properties

There are no missing values after preprocessing (we removed irrelevant columns and handled NaNs).

Models Used & Performance Metrics

The following six models were implemented for classification:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN) Classifier

Gaussian Naive Bayes Classifier

Random Forest (Ensemble)

XGBoost (Ensemble)

The evaluation metrics calculated for each model are:

Accuracy

AUC (Area Under the ROC Curve)

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

Comparison Table
ML Model Name	        Accuracy	  AUC	  Precision  	Recall	 F1 Score	 MCC
Logistic Regression	     0.96	      0.99  	0.95	     0.97	 0.96    	 0.92
Decision Tree	         0.94	      0.96	    0.93	     0.95	 0.94	     0.88
KNN	                     0.95	      0.97	    0.94	     0.96	 0.95	     0.9
Naive Bayes	             0.93	      0.95	    0.92	     0.94	 0.93	     0.86
Random Forest	         0.97	      0.99	    0.96	     0.98	 0.97	     0.94
XGBoost	                 0.97	      0.99	    0.96	     0.98	 0.97	     0.94



Observations About Model Performance


ML Model Name	           Observation about model performance
Logistic Regression	       Shows strong performance overall. It handles the dataset well with a good balance between precision and recall.
Decision Tree	           Works reasonably, but tends to overfit a bit. Accuracy is okay, but it’s less stable than ensemble methods.
KNN	                       Performs well on most cases. Slightly slower during prediction, but generally reliable.
Naive Bayes	               Gives decent results. Assumes that features are independent, which isn’t completely true, but still does a fair job.
Random Forest	           Very consistent and accurate. The ensemble approach reduces overfitting and improves stability.
XGBoost	                   Matches Random Forest in accuracy. Often faster in training and handles class differences better.
	
