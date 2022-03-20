# Keras GridSearch

Deep Classifier is an implementation of Scikeras's KerasClassifier to feed a Scikit's GridSearchCV, it is possible to compare through cross-validation several different hyperparams and layer architectures just providing them as a `param_grid`.
The typical architecture parameter will be a `dict` that contains as keys `Layers`, `ActivationFunctions` and `Neurons`, the values are arrays that contain the type of layer (`Dense` or `LSTM` for now), the type of activation functions (all supported by Keras) and the number of neurons (`int`). 
A possible `param_grid` configuration can look like the following dictionary:

`param_grid = {
    "optimizer__learning_rate": [0.00001, 0.1],
    "model__architecture": [
        {
            "Layers": ["Dense", "Dense"],
            "ActivationFunctions": ["relu", "sigmoid"],
            "Neurons": [20, 1],
        },
        {
            "Layers": ["Dense", "Dense", "Dense"],
            "ActivationFunctions": ["relu", "relu", "sigmoid"],
            "Neurons": [50, 20, 1],
        }
        ]
`
Here the idea is to test which one of these two architectures will perform better, also with two possible learning rates.

An example is provided with credit risk data in `Classification.ipynb`.

## How to Use

*To avoid all the prints in the console, change verbose to 0 in GridSearchCV and in DeepClassifier

1) Install requirements with `pip install -r requirements.txt`

2) Instantiate a base model of `Deep Classifier` with the default params that you defined, it is possible to add more params to Class `Deep Classifer`, this can be done in `DeepClassifier.py`

3) Provide a `param_grid` with different architectures and hyperparams that will be tested

4) Obtain the results
