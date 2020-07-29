from __future__ import annotations
from typing import Dict, List, Optional
from chess import Move, Board, BLACK
import numpy as np

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


class TreeNode:
    children: Dict[Move, TreeNode]

    def __init__(self, board: Board):
        self.state = State(board)
        self.W_state_val = 0
        self.N_num_visits = 0
        self.P_init_policy = None
        self.children = {}

    def get_Q(self):
        """
        Returns total value / num_visits.
        :return: W/N or expected win at the current state.
        """
        return self.W_state_val / max(self.N_num_visits, 1)

    def __hash__(self):
        return self.state.__hash__()

    def __eq__(self, other: TreeNode):
        return self.state.__eq__(other.state)



class State:
    def __init__(self, board: Board):
        self.board = board

    def transition_state(self, action: Move) -> TreeNode:
        board = self.board.copy()
        board.push(action)
        # new_state = State(board)
        return TreeNode(board)

    def get_actions(self) -> List[Move]:
        return list(self.board.legal_moves)

    def get_representation(self) -> np.ndarray:
        board = self.board
        state = np.zeros(64, np.uint8)
        fen = board.fen()
        str_i = 0
        i = 0
        piece_counter = 0

        while piece_counter < 64:
            c = fen[str_i]
            if c.isnumeric():
                for j in range(i, i + int(c)):
                    state[j] = 0
                piece_counter += int(c)

                str_i += 1
                i += int(c)
            elif c in FEN_MAP:
                state[i] = FEN_MAP[c]

                str_i += 1
                piece_counter += 1
                i += 1
            else:
                # Do nothing
                str_i += 1

        bit3 = (state >> 3) & 1
        bit2 = (state >> 2) & 1
        bit1 = (state >> 1) & 1
        bit0 = state & 1
        if board.turn == BLACK:
            turn_array = np.zeros(64, np.uint8)
        else:
            turn_array = np.ones(64, np.uint8)

        return np.array([bit3, bit2, bit1, bit0, turn_array])

    def serialise(self) -> str:
        to_string = self.get_representation()
        s = ""
        for array in to_string:
            for num in array:
                s += f'{num},'
            s = s[:-1] + '\n'
        return s

    def is_end(self) -> Optional[int]:
        if self.board.is_game_over():
            reward = {"1-0": 2, "1/2-1/2": 0, "0-1": -2}[self.board.result()]
            if self.board.turn == BLACK:
                return -reward
            return reward
        return None

    def get_policy(self, action: Move) -> float:
        # get 64 * first square, + 0-63
        return 0.3

    def __hash__(self):
        return (str(self.board) + str(self.board.halfmove_clock)).__hash__()

    def __eq__(self, other: State):
        return self.__hash__() == other.__hash__()