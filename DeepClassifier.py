from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Flatten,
    Input,
)
import numpy as np
from sklearn.utils import class_weight
from scikeras.wrappers import KerasClassifier


class DeepClassifier(KerasClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        train_ratio = self.train_ratio
        val_ratio = self.val_ratio
        batch_size = self.batch_size_custom
        kwargs["batch_size"] = batch_size
        kwargs["validation_split"] = val_ratio
        if self.balance_class_weights == True:
            weights = class_weight.compute_class_weight(class_weight='balanced',
                                                        classes=np.unique(y),
                                                        y=y)
            return super().fit(X=X, y=y, class_weight=dict(zip(np.unique(y),weights)), **kwargs)
        else:
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
    balance_class_weights,
):
    X_shape_0, X_shape_1, X_shape_2 = meta["X_shape_"]
    input_shape = (X_shape_1, X_shape_2)

    model = Sequential()
    model.train_ratio = train_ratio
    model.val_ratio = val_ratio
    model.batch_size = batch_size_custom
    model.balance_class_weights = balance_class_weights
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


