import os
import sys
import pandas as pd
from sklearn import datasets
import tensorflow as tf
from tensorflow import GradientTape
import pandas as pd
from sklearn.metrics import make_scorer
from tensorflow.keras import optimizers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Flatten,
    Input,
)
from scikeras.wrappers import KerasRegressor, KerasClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit


class DeepClassifier(KerasClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        train_ratio = self.train_ratio
        val_ratio = self.val_ratio
        batch_size = self.batch_size_custom
        kwargs["batch_size"] = batch_size
        kwargs["validation_split"] = val_ratio

        return super().fit(X=X, y=y, **kwargs)

    def predict(self, X, **kwargs):
        batch_size = self.batch_size_custom
        kwargs["batch_size"] = batch_size
        return super().predict(X=X, **kwargs)


def create_classifier(
    meta,
    architecture,
    train_ratio,
    val_ratio,
    batch_size_custom,
):
    X_shape_0, X_shape_1, X_shape_2 = meta["X_shape_"]
    input_shape = (X_shape_1, X_shape_2)

    model = Sequential()
    model.train_ratio = train_ratio
    model.val_ratio = val_ratio
    model.batch_size = batch_size_custom

    architecture_layers = architecture["Layers"]
    architecture_activation_functions = architecture["ActivationFunctions"]
    architecture_neurons = architecture["Neurons"]

    if len(architecture_layers) != len(architecture_activation_functions) or len(
        architecture_layers
    ) != len(architecture_neurons):
        raise Exception("Architecture has inconsistent attributes")

    for layer_number in range(len(architecture_layers)):

        layer_name = architecture_layers[layer_number]
        layer_activation_function = architecture_activation_functions[layer_number]
        layer_neurons = architecture_neurons[layer_number]

        if layer_name == "Dense":

            if layer_number == 0:
                model.add(
                    Input(
                        shape=X_shape_1,
                    )
                )

            model.add(Dense(layer_neurons, activation=layer_activation_function))

        if layer_name == "LSTM":
            model.add(
                LSTM(
                    layer_neurons,
                    activation=layer_activation_function,
                    input_shape=input_shape,
                )
            )

        if layer_name == "LSTM_Stacked":
            model.add(
                LSTM(
                    layer_neurons[0],
                    activation=layer_activation_function[0],
                    input_shape=input_shape,
                    return_sequences=True,
                )
            )

            model.add(LSTM(layer_neurons[1], activation=layer_activation_function[1]))

        if layer_name == "LSTM_Stateful":
            model.add(
                LSTM(
                    layer_neurons,
                    stateful=True,
                    activation=layer_activation_function,
                    batch_input_shape=(model.batch_size, X_shape_1, X_shape_2),
                )
            )

    return model


def predefined_split(X, batch_size_custom, train_ratio, val_ratio):
    """
    Predefines the split of training and validation set to be later used in the grid search
    """

    X = X.values.reshape((X.shape[0], X.shape[1], 1))

    multiple_train = int(((X.shape[0] * train_ratio) // batch_size_custom))
    multiple_val = int(((X.shape[0] * val_ratio) // batch_size_custom))

    rows_train = multiple_train * batch_size_custom
    rows_val = multiple_val * batch_size_custom

    train_list = [-1] * rows_train
    val_list = [0] * rows_val
    fold = train_list + val_list

    split = PredefinedSplit(test_fold=fold)
    return split


