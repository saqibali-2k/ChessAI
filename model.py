import torch
import numpy as np
from typing import Union

MODEL_PATH = "./models/v"

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


class ValuePolicyNet(torch.nn.Module):
    def __init__(self):
        super(ValuePolicyNet, self).__init__()
        self.conv_layer_1 = torch.nn.Conv2d(5, 256, 3)  # input size (8, 8, 5)
        self.batch_norm = torch.nn.BatchNorm2d(256, affine=False)
        self.relu_activation = torch.nn.ReLU()
        self.conv_layer_2 = torch.nn.Conv2d(256, 256, 3)  # input size (6, 6, 256) , (4 , 4, 256)
        self.conv_layer_3 = torch.nn.Conv2d(256, 256, 2)  # input size (2, 2, 256), output: (1, 1, 256)
        # self.conv_layer_policy = torch.nn.Conv2d(256, 2, 1)  input (1, 1, 256), output: (1, 1, 2)
        # Reduces features too much, might be good for ResNet though
        self.softmax_act = torch.nn.Softmax(dim=1)
        self.fc_layer_policy = torch.nn.Linear(256, 4096)
        self.tan_act = torch.nn.Tanh()
        self.fc_layer_value1 = torch.nn.Linear(256, 256)
        self.fc_layer_value2 = torch.nn.Linear(256, 1)

    def forward(self, states: Union[torch.tensor, np.ndarray], valid_moves: Union[torch.tensor, np.ndarray]):
        reshaped_states = torch.reshape(torch.tensor(states, dtype=torch.float32), (states.shape[0], 5, 8, 8))

        block1 = self.conv_layer_1(reshaped_states)
        block1 = self.batch_norm(block1)
        block1 = self.relu_activation(block1)

        block2 = self.conv_layer_2(block1)
        block2 = self.batch_norm(block2)
        block2 = self.relu_activation(block2)

        block3 = self.conv_layer_2(block2)
        block3 = self.batch_norm(block3)
        block3 = self.relu_activation(block3)

        block4 = self.conv_layer_3(block3)
        block4 = self.batch_norm(block4)
        block4 = self.relu_activation(block4)

        value = self.fc_layer_value1(torch.reshape(block4, (block4.shape[0], -1)))
        value = self.relu_activation(value)
        value = self.fc_layer_value2(value)
        value = self.tan_act(value)

        policy = self.fc_layer_policy(torch.reshape(block4, (block4.shape[0], -1)))
        policy = policy * torch.tensor(valid_moves, dtype=torch.float32)
        policy = self.softmax_act(policy)

        return value, policy


class CNNModel:

    def __init__(self, model_num: int):

        self.model_num = model_num

        self.model = ValuePolicyNet()

    def load_model(self):

        self.model = torch.load(MODEL_PATH + str(self.model_num))

    def load_weights(self):
        torch.save(self.model.state_dict(), MODEL_PATH + str(self.model_num))

    def save_weights(self, best=False):
        if best:
            torch.save(self.model.state_dict(), MODEL_PATH + 'best')
            return
        torch.save(self.model.state_dict(), MODEL_PATH + str(self.model_num))

    def train_model(self, inputs: np.ndarray, valids: np.ndarray, wins_loss: np.ndarray, improved_policies: np.ndarray):
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6, momentum=0.9, weight_decay=1e-4)

        for i in range(135):
            value_pred, policy_pred = self.model(inputs, valids)
            # log(0) returns NAN, so set NAN to 0
            log_policy = torch.tensor(improved_policies, dtype=torch.float32
                                      ).mm(torch.log(policy_pred).transpose(0, 1)).sum()
            log_policy[log_policy != log_policy] = 0

            loss = (value_pred - torch.tensor(wins_loss, dtype=torch.float32)).pow(2).sum() - log_policy

            if i % 50 == 0:
                print(f'Iteration {i}, loss: {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.save_weights()
        return loss.item()

    def evaluate(self, states, valids):
        self.model.eval()
        value, policy = self.model(torch.tensor(states, dtype=torch.float32), torch.tensor(valids, dtype=torch.float32))
        return policy, value

    def save_model(self, best=False):
        if best:
            torch.save(self.model, MODEL_PATH + "BEST")
            return
        torch.save(self.model, MODEL_PATH + str(self.model_num))
