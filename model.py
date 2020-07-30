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

    def __init__(self, model_num: int):
        self.model_num = model_num

        inputs = keras.Input(shape=(5, 64))
        valids = keras.Input(shape=(1, 4096))
        x = keras.layers.Reshape((8, 8, 5))(inputs)
        x = self._conv_block(x, 256, 3)
        x = self._conv_block(x, 256, 3)
        x = self._conv_block(x, 256, 3)
        x = self._conv_block(x, 256, 2)

        value = self._value_head(x)
        policy = self._policy_head(x, valids)

        self.model = keras.Model(inputs=[inputs, valids], outputs=[value, policy])

    def load_model(self, path):
        self.model = keras.models.load_model(path)

    def train_model(self, inputs, valids, wins_loss, improved_policies):
        self.model.compile(optimizer='adam',
                           loss=['mean_squared_error', 'categorical_crossentropy'],
                           loss_weights=[1.0, 1.0]
                           )

        self.model.fit([inputs, valids],
                       [wins_loss, improved_policies],
                       epochs=10
                       )

        keras.models.save_model(self.model, MODEL_PATH + str(self.model_num))

    def _policy_head(self, inputs, valids):
        x = self._conv_block(inputs, 2, 1)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(4096, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))(x)
        x = keras.layers.Activation('sigmoid')(x)
        x = keras.layers.multiply([x, valids])
        return x

    def _value_head(self, inputs):
        x = self._conv_block(inputs, 1, 1)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation='relu', kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))(x)
        x = keras.layers.Dense(1, activation='tanh')(x)
        return x

    def _conv_block(self, inputs, filters, kernel):
        x = keras.layers.Conv2D(filters, (kernel, kernel), kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))(inputs)
        x = keras.layers.BatchNormalization(axis=3)(x)
        x = keras.layers.Activation("relu")(x)
        return x

    def evaluate(self, states, valids):
        value, policy = self.model.predict([states, valids])
        return policy, value

    def get_data_from_tree(self, root):
        self.get_data_from_tree(root)

    def _format(self, board: chess.Board) -> np.ndarray:
        pass

    def get_policy(self):
        pass
