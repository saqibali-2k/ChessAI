from chess import Board, Move
from TreeNode import TreeNode, State
import numpy as np
from model import CNNModel
from typing import Union, Tuple, Optional
from random import choices

C_PUCT = 1.0
LETTER_MAP = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
NUMBER_MAP = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7}


class MonteCarloTS:
    curr: TreeNode
    root: TreeNode

    def __init__(self, initial_state, neural_net: CNNModel):
        self.root = TreeNode(initial_state)
        self.root.N_num_visits += 1
        self.curr = self.root
        self.visited = set()
        self.nnet = neural_net

    def get_best_action(self, node: TreeNode, training: bool) -> Optional[Move]:
        if not training:
            max_N, best = -float("inf"), None
            for move in node.children:
                n = node.children[move].N_num_visits
                if n > max_N:
                    max_N = n
                    best = node.children[move]
            return best
        else:
            move_lst, prob = self.get_improved_policy(node)
            return choices(move_lst, weights=prob)[0]

    def search(self, training=True):
        for _ in range(40):
            self._search(self.curr)
        best = self.get_best_action(self.curr, training=training)
        self.curr = self.curr.children[best]
        return best

    def enemy_move(self, move: Move):
        if move in self.curr:
            self.curr = self.curr.children[move]
            if self.curr.N_num_visits == 0:
                self.curr.N_num_visits += 1
        else:
            # we should be searching often enough so that this scenario does not occur, exploration parameter may need
            # to be adjusted
            raise Exception
            node = self.curr.state.transition_state(move)
            self.curr.children[move] = node
            self.curr = node
            if self.curr.N_num_visits == 0:
                self.curr.N_num_visits += 1

    def _search(self, curr: TreeNode):
        reward = curr.state.is_end()
        if reward is not None:
            return -1 * reward

        if curr not in self.visited:
            curr.P_init_policy, value = self.feed_network(curr)
            curr.W_state_val = value
            curr.N_num_visits += 1
            self.visited.add(curr)
            return -1 * value
        else:
            selected_child, max_u = None, -float("inf")
            poss_actions = curr.state.get_actions()
            sum = np.sqrt(curr.N_num_visits - 1)

            for action in curr.state.get_actions():
                if action in curr.children:
                    node = curr.children[action]
                    u = node.get_Q() + C_PUCT * curr.state.get_policy(action) * (sum / (1 + node.N_num_visits))
                else:
                    # initialize any non explored nodes at this point (with W = 0 and N = 0)
                    # But don't add them to visited nodes
                    node = curr.state.transition_state(action)
                    curr.children[action] = node
                    u = node.get_Q() + C_PUCT * curr.state.get_policy(action) * (sum / (1 + node.N_num_visits))

                if u > max_u:
                    u = max_u
                    selected_child = node
            value = self._search(selected_child)
            curr.N_num_visits += 1
            curr.W_state_val += value
            return -value

    def get_improved_policy(self, curr: TreeNode, include_empty_spots: bool = False) -> Union[
        Tuple[list, list], np.ndarray]:
        array = np.zeros(4096)

        if curr.N_num_visits > 1:
            sum = curr.N_num_visits - 1
        else:
            # We should not call on nodes that don't have explored children
            raise ZeroDivisionError

        move_lst = []
        probab = []
        for move in curr.children:
            policy = curr.children[move].N_num_visits / sum
            move_lst += [move]
            probab += [policy]
            array[self.sans_to_index(move.from_square, move.to_square)] = policy

        if include_empty_spots:
            return array
        return move_lst, policy

    def sans_to_index(self, from_square: str, to_square: str):
        index1 = LETTER_MAP[from_square[0]] * 8 + NUMBER_MAP[from_square[1]]
        index2 = LETTER_MAP[to_square[0]] * 8 + NUMBER_MAP[to_square[1]]
        return index1 * 64 + index2

    def get_policy(self, node: TreeNode, action: Move) -> float:
        # get 64 * first square, + 0-63
        return node.P_init_policy[self.sans_to_index(action.from_square, action.to_square)]

    def feed_network(self, curr: TreeNode) -> tuple:
        pass
