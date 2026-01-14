# Velum
A unified machine learning library combining classical methods and deep learning architectures under a consistent API. Built on PyTorch and scikit-learn.

## Motivation
Often, it can be very frustrating to build out models from scratch and attempt to combine models at once. 

I built Velum for a few reasons:

- **Learning By Building**
  - As a data science and mathematics student, I had the theory down, but wanted deeper fluency with the actual code. The best way to understand an algorithm is by implementing it, breaking it, and fixing it.

- **One Interface Overseeing Many Models**
  - Switching between scikit-learn, raw PyTorch, and various other libraries means constantly context-switching between different APIs. Velum will provide a consistent interface whether you're fitting a random forest or training a transformer.
 
- **A Growing Project**
  - Rather than building dozens of disconnected scripts, Velum is a single codebase I can keep extending to improve the API and reuse across future projects. This isn't meant to replace PyTorch or scikit-learn but to build on top of them.

## Overiew
Velum provides a growing collection of ML and deep learning models, from linear regression to Transformers, accessible through a single, consistent interface. Train models with minimal boilerplate, swap architectures easily, and focus on your data rather than implementation details.

### Available Models (To Be Built)

Classical ML (scikit-learn backend)
- Linear Regression
- Logistic Regression
- Random Forest
- Support Vector Machines
- K-Means Clustering
- PCA
- More coming...

Deep Learning (PyTorch backend)
- MLP (Multilayer Perceptron)
- CNN (Convolutional Neural Network)
- LSTM
- CNN-LSTM
- Transformer Encoder
- Autoencoder
- More coming...

I am also hoping to extend this further with RL and search specific models; this will be on hold for now.

## Installation
```
git clone https://github.com/yourusername/velum.git
cd velum
pip install -e .
```


## Contributing
Suggestions and feedback are welcome.

## License
MIT


