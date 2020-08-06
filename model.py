import torch
import numpy as np
from typing import Union
import tqdm

MODEL_PATH = "./models/v"

BATCH_SIZE = 128

EPOCHS = 20

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


class StatWrapper:

    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, x):
        self.total += x
        self.count += 1

    def __repr__(self):
        avg = self.total / self.count
        return f'{avg: .2f}'


class ValuePolicyNet(torch.nn.Module):
    def __init__(self, num_res_blocks: int=3):
        super(ValuePolicyNet, self).__init__()
        self.conv_layer_1 = torch.nn.Conv2d(5, 256, 3, padding=1)  # input size (5, 8, 8)
        self.batch_norm1 = torch.nn.BatchNorm2d(256)
        self._num_res_blocks = num_res_blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks += [torch.nn.Conv2d(256, 256, 3, padding=1)]
            res_blocks += [torch.nn.BatchNorm2d(256)]
            res_blocks += [torch.nn.Conv2d(256, 256, 3, padding=1)]
            res_blocks += [torch.nn.BatchNorm2d(256)]
            # input: (256, 8 , 8), output: (256, 8, 8)

        self.res_blocks = torch.nn.ModuleList(res_blocks)


        # self.batch_norm2 = torch.nn.BatchNorm2d(256)
        # self.batch_norm3 = torch.nn.BatchNorm2d(256)
        # self.batch_norm4 = torch.nn.BatchNorm2d(256)
        # self.batch_norm5 = torch.nn.BatchNorm2d(256)
        # self.batch_norm6 = torch.nn.BatchNorm2d(256)
        # self.batch_norm7 = torch.nn.BatchNorm2d(256)
        self.relu_activation = torch.nn.functional.relu
        # self.conv_layer_2 = torch.nn.Conv2d(256, 256, 3, padding=1)  # input size (256, 8, 8)
        # self.conv_layer_3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        # self.conv_layer_4 = torch.nn.Conv2d(256, 256, 3, padding=1)  # input (256, 8 , 8), output:(256, 8, 8)
        # self.conv_layer_5 = torch.nn.Conv2d(256, 256, 3, padding=1)  # input (256, 8 , 8), output:(256, 8, 8)
        # self.conv_layer_6 = torch.nn.Conv2d(256, 256, 3, padding=1)  # input size (256, 8, 8), output: (256, 8, 8)
        # self.conv_layer_7 = torch.nn.Conv2d(256, 256, 3, padding=1)  # input size (256, 8, 8), output: (256, 8, 8)
        # self.conv_layer_reduc = torch.nn.Conv2d(256, 2, 1)  # input (256, 8, 8), output: (1, 1, 2)

        # Reduces features too much, might be good for ResNet though

        self.fc_layer_policy = torch.nn.Linear(16384, 4096)
        #128
        self.fc_layer_value1 = torch.nn.Linear(16384, 256)
        self.fc_layer_value2 = torch.nn.Linear(256, 1)

    def forward(self, states: Union[torch.tensor, np.ndarray], valid_moves: Union[torch.tensor, np.ndarray]):

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        reshaped_states = torch.reshape(torch.tensor(states, dtype=torch.float32, device=device),
                                        (states.shape[0], 5, 8, 8))

        block1 = self.conv_layer_1(reshaped_states)
        block1 = self.batch_norm1(block1)
        block1 = self.relu_activation(block1)

        res_in = block1
        for i in range(self._num_res_blocks):
            res_out = self.res_blocks[i * 4 + 0](res_in)  # Conv
            res_out = self.res_blocks[i * 4 + 1](res_out)  # BatchNorm
            res_out = self.relu_activation(res_out)

            res_out = self.res_blocks[i * 4 + 2](res_out)  # Conv
            res_out = self.res_blocks[i * 4 + 3](res_out)  # BatchNorm
            res_out = torch.add(res_out, res_in)
            res_out = self.relu_activation(res_out)
            res_in = res_out
        res_out = res_in

        value = self.fc_layer_value1(torch.reshape(res_out, (res_out.shape[0], -1)))
        value = self.relu_activation(value)
        value = self.fc_layer_value2(value)
        value = torch.tanh(value)

        policy = self.fc_layer_policy(torch.reshape(res_out, (res_out.shape[0], -1)))
        policy = policy * torch.tensor(valid_moves, dtype=torch.float32, device=device)
        policy = torch.nn.functional.log_softmax(policy, dim=1)

        return value, policy


class CNNModel:

    def __init__(self, model_num: int):

        self.model_num = model_num

        self.model = ValuePolicyNet(10)

        if torch.cuda.is_available():
            self.model.to(torch.device("cuda:0"))
        else:
            self.model.to(torch.device("cpu"))

    def load_model(self):

        self.model = torch.load(MODEL_PATH + str(self.model_num))

    def load_weights(self, path: str=None):
        if path is None:
            self.model.load_state_dict(torch.load(MODEL_PATH + str(self.model_num)))
        else:
            self.model.load_state_dict(torch.load(path))

    def save_weights(self, best=False):
        if best:
            torch.save(self.model.state_dict(), MODEL_PATH + 'best')
            return
        torch.save(self.model.state_dict(), MODEL_PATH + str(self.model_num))

    def train_model(self, inputs: np.ndarray, valids: np.ndarray, wins_loss: np.ndarray, improved_policies: np.ndarray):
        self.model.train()

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-4)

        for i in range(EPOCHS):

            num_batches = wins_loss.shape[0]//BATCH_SIZE
            p_bar = tqdm.tqdm(range(num_batches), desc=f"Training Network Epoch-{i}")

            rand_indices = torch.randperm(wins_loss.shape[0])
            wins_loss = wins_loss[rand_indices]
            improved_policies = improved_policies[rand_indices]
            inputs = inputs[rand_indices]
            valids = valids[rand_indices]

            # Stats to report
            avg_loss = StatWrapper()
            avg_policy_loss = StatWrapper()
            avg_value_loss = StatWrapper()

            for _ in p_bar:

                # Make batch
                end_index = min((_ + 1) * BATCH_SIZE, wins_loss.shape[0])

                value_expected_batch = wins_loss[_ * BATCH_SIZE: end_index]
                policy_expected_batch = improved_policies[_ * BATCH_SIZE: end_index, :]
                states_batch = inputs[_ * BATCH_SIZE: end_index, :, :]
                valids_batch = valids[_ * BATCH_SIZE: end_index, :]

                policy_expected_batch = torch.tensor(policy_expected_batch, dtype=torch.float32, device=device)
                value_expected_batch = torch.tensor(value_expected_batch, dtype=torch.float32, device=device)

                # Get outputs
                value_pred, policy_pred = self.model(states_batch, valids_batch)
                value_pred = torch.flatten(value_pred)

                # Calculate losses
                policy_loss = -1 * policy_expected_batch.mm(policy_pred.to(device).transpose(0, 1)).sum()
                value_loss = (value_pred.to(device) - value_expected_batch).pow(2).sum()

                loss = policy_loss + value_loss

                # Update stats
                avg_loss.update(loss.item())
                avg_value_loss.update(value_loss.item())
                avg_policy_loss.update(policy_loss.item())
                p_bar.set_postfix(loss=avg_loss, policy_loss=avg_policy_loss, value_loss=avg_value_loss)

                # Perform back prop and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        del policy_expected_batch, value_expected_batch
        del value_pred, policy_pred

        self.save_weights()

    def evaluate(self, states, valids):
        self.model.eval()
        with torch.no_grad():
            value, policy = self.model(states, valids)
            return policy.cpu(), value.cpu()

    def save_model(self, best=False):
        if best:
            torch.save(self.model, MODEL_PATH + "BEST")
            return
        torch.save(self.model, MODEL_PATH + str(self.model_num))

    def delete_model(self):
        del self.model
