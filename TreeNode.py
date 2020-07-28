from __future__ import annotations
from typing import Dict


class TreeNode:
    children: dict

    def __init__(self, state):
        self.state = state
        self.W_state_val = 0
        self.N_num_visits = 0
        self.P_init_policy = None
        self.children = {}

    def get_Q(self):
        """
        Returns total value / num_visits.
        :return: W/N or expected win at the current state.
        """
        return self.W_state_val / self.N_num_visits
