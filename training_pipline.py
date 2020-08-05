import chess
from MonteCarloTS import MonteCarloTS
from model import CNNModel
import numpy as np
import torch.multiprocessing as mp
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='./debug.log',
                    level=logging.DEBUG, filemode='w')

TURN_CUTOFF = 180
SELF_GAMES = 80
NUM_TRAINS = 400
BOT_GAMES = 20

CPU_COUNT = max(mp.cpu_count() - 1, 1)


def training_pipeline():
    while True:
        mode = input("mode?")
        if mode == "continue" or mode == "new":
            break
        print("Not recognized")
    model_num = 1
    best_model_num = 0
    best_model = CNNModel(best_model_num)

    if mode == "continue":
        best_model.load_weights(path="models/vbest")
    best_model.save_weights()

    for _ in range(NUM_TRAINS):
        logging.info(f"Training iter {_}: self-play started")
        states, valids, improved_policy, win_loss = self_play(_, best_model_num)

        contender = CNNModel(model_num)
        contender.load_weights(path="models/vbest")

        logging.info(f"Training iter {_}: contender model training started")
        contender.train_model(np.array(states, np.float), np.array(valids, np.float), np.array(win_loss, np.float),
                             np.array(improved_policy, np.float))

        logging.info(f"Training iter {_}: contender model training finished")
        contender_wins, best_wins = bot_fight(_, best_model.model_num, contender.model_num)

        win_ratio = contender_wins / max(best_wins, 1)
        if win_ratio >= 4/3:
            best_model = contender
            best_model_num = contender.model_num
        logging.info(f'Training iter {_}: best model: {best_model_num}, new model won {contender_wins}')
        print(f'Training iter {_}: best model: {best_model_num}, new model won {contender_wins}')
        best_model.save_weights(best=True)
        model_num += 1


def self_play(i, best_model_num):

    states, valids, improv_policy, win_loss = [], [], [], []

    pool = mp.Pool(CPU_COUNT)
    results_objs = [pool.apply_async(async_episode, args=(best_model_num,)) for _ in range(SELF_GAMES)]
    pool.close()

    p_bar = tqdm(range(len(results_objs)), desc=f"Self-play-{i}")

    for i in p_bar:
        result = results_objs[i]
        result = result.get()
        states += result[0]
        valids += result[1]
        improv_policy += result[2]
        win_loss += result[3]

    return states, valids, improv_policy, win_loss


def async_episode(best_model_num) -> tuple:
    valids, states, improv_policy, win_loss = [], [], [], []

    best_model = CNNModel(best_model_num)
    best_model.load_weights()

    board = chess.Board()
    mcts = MonteCarloTS(board.copy(), best_model)

    visited_nodes = []
    while not board.is_game_over() and board.fullmove_number < TURN_CUTOFF:
        visited_nodes.append(mcts.curr)
        move = mcts.search()
        board.push(move)

    reward_white = {"1-0": 1,
                    "1/2-1/2": 0,
                    "*": 0,
                    "0-1": -1}
    logging.info(f'finished game with {board.result()}')

    for node in visited_nodes:
        policy = mcts.get_improved_policy(node, include_empty_spots=True)
        z = reward_white[board.result()]
        if node.state.board.turn == chess.BLACK:
            z *= -1
        states.append(node.state.get_representation())
        valids.append(node.state.get_valid_vector())
        improv_policy.append(policy)
        win_loss.append(z)
    return states, valids, improv_policy, win_loss


def bot_fight(i, best_model_num, new_model_num) -> tuple:
    new_model_wins = 0
    best_model_wins = 0
    result_objs = []
    pool = mp.Pool(CPU_COUNT)
    for j in range(BOT_GAMES):
        result_objs += [pool.apply_async(async_arena, args=(j, best_model_num, new_model_num))]
    pool.close()

    p_bar = tqdm(range(len(result_objs)), desc=f"Bot battle-{i}")

    for i in p_bar:
        result = result_objs[i]
        result = result.get()
        new_model_wins += result[0]
        best_model_wins += result[1]

    return new_model_wins, best_model_wins


def async_arena(iteration, best_model_num, new_model_num):

    new_model_wins = 0
    best_model_wins = 0
    board = chess.Board()

    best_model = CNNModel(best_model_num)
    best_model.load_weights()
    new_model = CNNModel(new_model_num)
    new_model.load_weights()

    mcts_best = MonteCarloTS(chess.Board(), best_model)
    mcts_new = MonteCarloTS(chess.Board(), new_model)

    if iteration % 2 == 0:
        turns = {"best": chess.WHITE,
                 "new": chess.BLACK}
    else:
        turns = {"best": chess.BLACK,
                 "new": chess.WHITE}
    while not board.is_game_over() and board.fullmove_number < TURN_CUTOFF and not board.is_repetition(count=4):
        if turns["best"] == chess.WHITE:
            move = mcts_best.search(training=True)
            board.push(move)
            mcts_new.enemy_move(move)

            move = mcts_new.search(training=True)
            if move is None:
                break
            board.push(move)
            mcts_best.enemy_move(move)
        else:
            move = mcts_new.search(training=True)
            board.push(move)
            mcts_best.enemy_move(move)

            move = mcts_best.search(training=True)
            if move is None:
                break
            board.push(move)
            mcts_new.enemy_move(move)
    s = board.result()

    if s == "1-0" and turns["new"] == chess.WHITE:
        new_model_wins += 1
    elif s == "0-1" and turns["new"] == chess.BLACK:
        new_model_wins += 1
    elif s == "1-0" and turns["best"] == chess.WHITE:
        best_model_wins += 1
    elif s == "0-1" and turns["best"] == chess.BLACK:
        best_model_wins += 1

    if new_model_wins == 1:
        logging.info("new_model won")
    return new_model_wins, best_model_wins


if __name__ == "__main__":
    # To prevent unix leakages (prevents Error initialising CUDA)
    mp.set_start_method('spawn')
    training_pipeline()
