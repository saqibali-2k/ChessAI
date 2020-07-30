
import chess
from MonteCarloTS import MonteCarloTS
from model import CNNModel
import numpy as np

SELF_GAMES = 10
NUM_TRAINS = 15
BOT_GAMES = 10


def training_pipeline():
    model_num = 0
    best_model_num = 0
    best_model = CNNModel(model_num)
    for _ in range(NUM_TRAINS):
        valids, states, improved_policy, win_loss = [], [], [], []
        states, improved_policy, win_loss = self_play(best_model, states, valids, improved_policy, win_loss)
        contender = CNNModel(model_num + 1)

        contender.train_model(np.array(states, np.uint32), np.array(valids, np.float32), np.array(win_loss),
                              np.array(improved_policy))
        contender_wins = bot_fight(best_model, contender)
        if contender_wins >= np.ceil(BOT_GAMES * 0.55):
            best_model = contender
            best_model_num = model_num + 1
        print(f'best model: {best_model_num}, new model won {contender_wins}')
        model_num += 1


def self_play(best_model, states: list, valids: list, improv_policy: list, win_loss: list):
    for _ in range(SELF_GAMES):
        board = chess.Board()
        mcts = MonteCarloTS(board.copy(), best_model)
        while not board.is_game_over():
            move = mcts.search()
            board.push(move)

        reward_white = {"1-0": 1,
                        "1/2-1/2": 0,
                        "0-1": -1}

        print(f'finished game {_} with {board.result()}')

        for node in mcts.visited:
            policy = mcts.get_improved_policy(node, include_empty_spots=True)
            z = reward_white[board.result()]
            if node.state.board.turn == chess.BLACK:
                z *= -1
            states.append(node.state.get_representation())
            valids.append(node.state.get_valid_vector())
            improv_policy.append(policy)
            win_loss.append(z)

    return states, improv_policy, win_loss


def bot_fight(best_model, new_model) -> int:
    new_model_wins = 0
    mcts_best = MonteCarloTS(chess.Board(), best_model)
    mcts_new = MonteCarloTS(chess.Board(), new_model)
    for _ in range(BOT_GAMES):
        board = chess.Board()

        if _ % 2 == 0:
            turns = {"best": chess.WHITE,
                     "new": chess.BLACK}
        else:
            turns = {"best": chess.BLACK,
                     "new": chess.WHITE}

        while not board.is_game_over() and not board.is_seventyfive_moves() and not board.is_fivefold_repetition():
            if turns["best"] == chess.WHITE:
                move = mcts_best.search(training=False)
                board.push(move)
                mcts_new.enemy_move(move)

                move = mcts_new.search(training=False)
                if move is None:
                    break
                board.push(move)
                mcts_best.enemy_move(move)
            else:
                move = mcts_new.search(training=False)
                board.push(move)
                mcts_best.enemy_move(move)

                move = mcts_best.search(training=False)
                if move is None:
                    break
                board.push(move)
                mcts_new.enemy_move(move)

        s = board.result()
        if s == "1-0" and turns["new"] == chess.WHITE:
            new_model_wins += 1
        elif s == "0-1" and turns["new"] == chess.BLACK:
            new_model_wins += 1
        mcts_new.reset_tree()
        mcts_best.reset_tree()

    return new_model_wins


if __name__ == "__main__":
    training_pipeline()
