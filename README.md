# Keras GridSearch


## How to Use

*To avoid all the prints in the console, change verbose to 0 in GridSearchCV and in DeepClassifier

1) Install requirements with `pip install -r requirements.txt`

2) Instantiate a base model of `Deep Classifier` with the default params that you defined, it is possible to add more params to Class `Deep Classifer`, this can be done in `DeepClassifier.py`

3) Provide a `param_grid` with different architectures and hyperparams that will be tested

4) Obtain the results
