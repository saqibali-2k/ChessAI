from chess import Board, Move
from TreeNode import TreeNode, State, sans_to_index
import numpy as np
from model import CNNModel
from typing import Union, Tuple, Optional
from random import choices

NUM_SIMULATIONS = 50

C_PUCT = 1.0
LETTER_MAP = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
NUMBER_MAP = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7}


class MonteCarloTS:
    curr: TreeNode
    root: TreeNode

    def __init__(self, initial_state, neural_net: CNNModel):
        self.root = TreeNode(initial_state)
        self.curr = self.root
        self.visited = set()
        self.nnet = neural_net

    def get_best_action(self, node: TreeNode, training: bool) -> Optional[Move]:
        if not training:
            max_N, best = -float("inf"), None
            for move in node.state.get_actions():
                if move in node.children:
                    n = node.children[move].N_num_visits
                else:
                    n = 0
                if n > max_N:
                    max_N = n
                    best = move
            # print(best)
            return best
        else:
            move_lst, prob = self.get_improved_policy(node)
            if len(move_lst) != 0:
                best = choices(move_lst, weights=prob)
                best = best[0]
            else:
                best = None
            # print(best)
            return best

    def search(self, training=True):
        for _ in range(NUM_SIMULATIONS):
            self._search(self.curr)
        best = self.get_best_action(self.curr, training=training)
        if best is not None:
            if best not in self.curr.children:
                # Unreachable code
                raise RuntimeError
            self.curr = self.curr.children[best]
        return best

    def enemy_move(self, move: Move):
        if move in self.curr.children:
            self.curr = self.curr.children[move]
        else:
            # we let it happen
            node = self.curr.state.transition_state(move)
            self.curr.children[move] = node
            self.curr = node

    def _search(self, curr: TreeNode):
        reward = curr.state.is_end()
        if reward is not None:
            return -1 * reward

        if curr not in self.visited:
            curr.P_init_policy, value = self.feed_network(curr)
            curr.W_state_val = 0
            curr.N_num_visits = 0
            self.visited.add(curr)
            return -1 * value
        else:
            best_action, selected_child, max_u = None, None, -float("inf")
            sum_visits = curr.N_num_visits

            for action in curr.state.get_actions():
                if action in curr.children:
                    node = curr.children[action]
                    u = node.get_Q() + C_PUCT * curr.state.get_policy(action) * (np.sqrt(sum_visits) / (1 + node.N_num_visits))
                else:
                    # initialize any non explored nodes at this point (with W = 0 and N = 0)
                    # But don't add them to visited nodes

                    u = C_PUCT * curr.state.get_policy(action) * np.sqrt(sum_visits + 1e-8)  # to encourage exploring

                if u > max_u:
                    max_u = u
                    best_action = action

            if best_action in curr.children:
                selected_child = curr.children[best_action]
            else:
                selected_child = curr.state.transition_state(best_action)
                curr.children[best_action] = selected_child

            value = self._search(selected_child)
            curr.N_num_visits += 1
            curr.W_state_val += value
            return -value

    def get_improved_policy(self, curr: TreeNode, include_empty_spots: bool = False) -> Union[
        Tuple[list, list], np.ndarray]:
        array = np.zeros(4096)

        sum = max(curr.N_num_visits - 1, 1)

        move_lst = []
        probab = []
        for move in curr.children:
            policy = curr.children[move].N_num_visits / sum
            move_lst += [move]
            probab += [policy]
            array[sans_to_index(move.from_square, move.to_square)] = policy

        if include_empty_spots:
            return array
        return move_lst, probab

    def print_tree(self, root: TreeNode, space: int) -> None:
        print(" " * space, root.N_num_visits)
        for action in root.children:
            self.print_tree(root.children[action], space + 2)

    def get_policy(self, node: TreeNode, action: Move) -> float:
        # get 64 * first square, + 0-63
        return node.P_init_policy[sans_to_index(action.from_square, action.to_square)]

    def reset_tree(self):
        self.curr = self.root

    def feed_network(self, curr: TreeNode) -> tuple:
        valids = self.curr.state.get_valid_vector()
        policy, value = self.nnet.evaluate(np.array([curr.state.get_representation()]), np.array([valids]))
        # mask illegal moves and renormalize

        return policy[0], value.item()

