import chess
from MonteCarloTS import MonteCarloTS
from model import CNNModel
import numpy as np

NUM_GAMES = 10
NUM_TRAINS = 15


def training_pipline():
    model_num = 0
    best_model_num = 0
    best_model = CNNModel(model_num)
    for _ in range(NUM_TRAINS):
        states, improved_policy, win_loss = [], [], []
        states, improved_policy, win_loss = self_play(best_model, states, improved_policy, win_loss)
        contender = CNNModel(model_num + 1)
        contender.train_model(states, win_loss, improved_policy)
        contender_wins = bot_fight(best_model, contender)
        if contender_wins > np.ceil(NUM_TRAINS * 0.55):
            best_model = contender
            best_model = model_num + 1
        print(best_model_num)
        model_num += 1


def self_play(best_model, states: list, improv_policy: list, win_loss: list):
    for _ in range(NUM_GAMES):
        board = chess.Board()
        mcts = MonteCarloTS(board.copy(), best_model)
        while not board.is_game_over():
            move = mcts.search()
            board.push(move)

        reward_white = {"1-0": 1,
                        "1/2-1/2": 0,
                        "0-1": -1}
        mcts.print_tree(mcts.root, 0)
        print(f'finished game {_} with {board.result()}')

        for node in mcts.visited:
            if node.children != {}:
                for action in node.children:
                    child = node.children[action]
                    policy = mcts.get_improved_policy(node, include_empty_spots=True)
                    z = reward_white[board.result()]
                    if child.state.board.turn == chess.BLACK:
                        z *= -1
                    states.append(child.state.get_representation())
                    improv_policy.append(policy)
                    win_loss.append(z)

    return states, improv_policy, win_loss


def bot_fight(best_model, new_model) -> int:
    new_model_wins = 0
    for _ in range(25):
        board = chess.Board()
        if _ % 2 == 0:
            turns = {"best": chess.WHITE,
                     "new": chess.BLACK}
            mcts_best = MonteCarloTS(board.copy(), best_model)
            move = mcts_best.search(training=False)
            board.push(move)
            mcts_new = MonteCarloTS(board.copy(), new_model)
        else:
            turns = {"best": chess.BLACK,
                     "new": chess.WHITE}
            mcts_new = MonteCarloTS(board.copy(), best_model)
            move = mcts_new.search(training=False)
            board.push(move)
            mcts_best = MonteCarloTS(board.copy(), new_model)

        while not board.is_game_over():
            if turns["best"] == chess.WHITE:
                move = mcts_best.search(training=False)
                board.push(move)
                mcts_new.enemy_move(move)

                move = mcts_new.search(training=False)
                board.push(move)
                mcts_best.enemy_move(move)
            else:
                move = mcts_new.search(training=False)
                board.push(move)
                mcts_best.enemy_move(move)

                move = mcts_best.search(training=False)
                board.push(move)
                mcts_new.enemy_move(move)

        s = board.result()
        if s == "1-0" and turns["new"] == chess.WHITE:
            new_model_wins += 1
        elif s == "0-1" and turns["new"] == chess.BLACK:
            new_model_wins += 1

    return new_model_wins

if __name__ == "__main__":
    training_pipline()
