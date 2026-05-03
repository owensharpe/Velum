"""
Title: __init__.py
Author: Owen Sharpe
Description: Exports all classical ML models from the velum.classical subpackage.
"""

from velum.classical.linear_regression import LinearRegression
from velum.classical.logistic_regression import LogisticRegression
from velum.classical.decision_tree import DecisionTree
from velum.classical.random_forest import RandomForest
from velum.classical.knn import KNN
from velum.classical.svm import SVM
from velum.classical.naive_bayes import NaiveBayes
from velum.classical.gradient_boosting import GradientBoosting
from velum.classical.ridge_regression import RidgeRegression
from velum.classical.lasso_regression import LassoRegression
from velum.classical.elastic_net import ElasticNetRegression
from velum.classical.kmeans import KMeans
from velum.classical.pca import PCA
