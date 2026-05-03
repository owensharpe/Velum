"""
Title: __init__.py
Author: Owen Sharpe
Description: Basic init file; will tell Python that our folder is the package itself.
Velum is a unified machine learning library.
"""

__version__ = "0.1.0"

from velum.classical import (
    LinearRegression,
    LogisticRegression,
    DecisionTree,
    RandomForest,
    KNN,
    SVM,
    NaiveBayes,
    GradientBoosting,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression,
    KMeans,
    PCA,
)

from velum.deep import (
    MLP,
    AutoEncoder,
    LSTM,
    CNN,
    Transformer,
    TabTransformer,
)
