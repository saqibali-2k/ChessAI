from typing import Callable
from TreeNode import TreeNode
import numpy as np


class MonteCarloTS:
    def __init__(self, initial_state, neural_net):
        self.root = TreeNode(initial_state)
        self.curr = self.root
        self.visited = set()
        self.nnet = neural_net

    def get_best_action(self, node: TreeNode):
        Q = node.get_Q()
        P = node.P_init_policy

    def search(self):
        self._search(self.curr)

    def _search(self, curr: TreeNode):
        reward = curr.state.is_end()
        if reward is not None:
            return -1 * reward

        if curr not in self.visited:
            curr.P_init_policy, value = self.feed_network(curr)
            self.visited.add(curr)
            return -1 * value
        else:
            for action in curr.state.get_actions():
                # initialize any non explored nodes at this point (with W = 0 and N = 0)
                # But don't add them to visited nodes
                pass

    def feed_network(self, curr) -> tuple:
        pass


