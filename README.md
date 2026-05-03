# Velum
A unified machine learning library combining classical methods and deep learning architectures under a consistent API. Built on PyTorch and scikit-learn.

## Motivation
I built Velum for a few reasons:
 
- **One Interface Overseeing Many Models**
  - Switching between scikit-learn, raw PyTorch, and various other libraries means constantly context-switching between different APIs. Velum provides a consistent `fit` / `predict` / `score` interface whether you're fitting a random forest or training a transformer.
- **A Growing Project**
  - Rather than building dozens of disconnected scripts, Velum is a single codebase I can keep extending and reuse across future projects. This isn't meant to replace PyTorch or scikit-learn, but it builds on top of them.
- **Built Toward a Platform**
  - Velum is the foundation for a web-based ML platform where users can upload data, configure and train models through a chat interface, and get predictions back in real time with no code required.

## Overview
Velum provides a collection of classical ML and deep learning models accessible through a single, consistent interface. Train models with minimal boilerplate, swap architectures easily, and focus on your data rather than implementation details.

```python
from velum import RandomForest, MLP
 
# Classical Example
model = RandomForest(n_estimators=100)
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)
 
# Deep Example
model = MLP(hidden_layers=[128, 64], task='classification', epochs=50)
model.fit(X_train, y_train)
model.predict(X_test)
```

## Available Models
 
**Classical ML (scikit-learn backend)**
| Model | Class | Type |
|---|---|---|
| Linear Regression | `LinearRegression` | Regression |
| Ridge Regression | `RidgeRegression` | Regression |
| Lasso Regression | `LassoRegression` | Regression |
| Elastic Net | `ElasticNetRegression` | Regression |
| Logistic Regression | `LogisticRegression` | Classification |
| Decision Tree | `DecisionTree` | Classification |
| Random Forest | `RandomForest` | Classification |
| Gradient Boosting | `GradientBoosting` | Classification |
| Support Vector Machine | `SVM` | Classification |
| K-Nearest Neighbors | `KNN` | Classification |
| Naive Bayes | `NaiveBayes` | Classification |
| K-Means | `KMeans` | Clustering |
| PCA | `PCA` | Dimensionality Reduction |
 
**Deep Learning (PyTorch backend)**
| Model | Class | Type |
|---|---|---|
| Multilayer Perceptron | `MLP` | Classification / Regression |
| Autoencoder | `AutoEncoder` | Unsupervised / Reconstruction |
| LSTM | `LSTM` | Sequential / Time Series |
| CNN | `CNN` | Image / Sequence |
| Transformer | `Transformer` | Sequential / Text |
| Tabular Transformer | `TabTransformer` | Tabular Classification / Regression |

I am also hoping to extend this further with RL and search specific models; this will be on hold for now.

## Installation
```bash
git clone https://github.com/yourusername/velum.git
cd velum
pip install -e .
```

## Requirements
- Python 3.9+
- PyTorch 2.0+
- scikit-learn 1.0+
- numpy
- pandas
- matplotlib, seaborn, plotly
- fastapi, uvicorn (for the web platform)

## GPU Support (CUDA)
The default installation uses CPU-only PyTorch. For GPU support, install PyTorch with CUDA separately before installing Velum:
```bash
# Example for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
Check [pytorch.org](https://pytorch.org/get-started/locally/) for the right command for your CUDA version.
 
## Contributing
Suggestions and feedback are welcome.
 
## License
MIT


