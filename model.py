import tensorflow as tf
import tensorflow.keras as keras
import chess
import numpy as np

MODEL_PATH = "models/v"

FEN_MAP = {"K": 6,
           "Q": 5,
           "B": 4,
           "N": 3,
           "R": 2,
           "P": 1,
           "k": -6,
           "q": -5,
           "b": -4,
           "n": -3,
           "r": -2,
           "p": -1}


class ChessModel:

    # def __init__(self, path: str):
    #     self.model = tf.keras.models.load_model(path)

    def _format(self, board: chess.Board) -> np.ndarray:
        raise NotImplementedError

    def get_policy(self):
        raise NotImplementedError

    # def evaluate(self, board: chess.Board) -> np.ndarray:
    #     array = self._format(board)
    #     return np.array(self.model(array))


class CNNModel(ChessModel):

    def __init__(self, model_num):
        self.model_num = model_num

        inputs = keras.Input(shape=(8, 8, 5))
        x = self._conv_block(inputs, 256, 3)
        x = self._conv_block(x, 256, 3)
        x = self._conv_block(x, 256, 3)
        x = self._conv_block(x, 256, 2)

        value = self._value_head(x)
        policy = self._policy_head(x)

        self.model = keras.Model(inputs=[inputs], outputs=[value, policy])

    def train_model(self, input, wins_loss, improved_policies):
        self.model.compile(optimizer='adam',
                           loss={"value": keras.losses.MeanSquaredError,
                                 "policy": tf.nn.sigmoid_cross_entropy_with_logits},
                           loss_weights=[1.0, 1.0]
                           )

        self.model.fit(input,
                       {"value": wins_loss,
                        "policy": improved_policies},
                       epochs=10
                       )

        keras.models.save_model(self.model, MODEL_PATH + str(self.model_num))

    def _policy_head(self, inputs):
        x = self._conv_block(inputs, 2, 1)
        x = keras.layers.Dense(4096, activation='sigmoid')(x)
        return x

    def _value_head(self, inputs):
        x = self._conv_block(inputs, 1, 1)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(1, activation='tanh')(x)
        return x

    def _conv_block(self, inputs, filters, kernel):
        x = keras.layers.Conv2D(filters, (kernel, kernel))(inputs)
        x = keras.layers.BatchNormalization(axis=3)(x)
        x = keras.layers.Activation("relu")(x)
        return x

    def get_data_from_tree(self, root):
        self.get_data_from_tree(root)

    def _format(self, board: chess.Board) -> np.ndarray:
        pass

    def get_policy(self):
        pass
