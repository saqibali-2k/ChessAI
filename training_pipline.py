import chess
from MonteCarloTS import MonteCarloTS
from model import CNNModel
import numpy as np
import multiprocessing as mp


SELF_GAMES = 30
NUM_TRAINS = 15
BOT_GAMES = 10

CPU_COUNT = mp.cpu_count() - 1


def training_pipeline():
    model_num = 1
    best_model_num = 0
    best_model = CNNModel(best_model_num)
    best_model.save_model()
    for _ in range(NUM_TRAINS):
        states, valids, improved_policy, win_loss = self_play(best_model_num)

        contender = CNNModel(model_num)

        contender.train_model(np.array(states, np.uint32), np.array(valids, np.float32), np.array(win_loss),
                              np.array(improved_policy))
        contender_wins = bot_fight(best_model.model_num, contender.model_num)

        if contender_wins >= np.ceil(BOT_GAMES * 0.55):
            best_model = contender
            best_model_num = model_num + 1
        print(f'best model: {best_model_num}, new model won {contender_wins}')
        best_model.save_model(best=True)
        model_num += 1


def self_play(best_model_num):

    states, valids, improv_policy, win_loss = [], [], [], []

    pool = mp.Pool(CPU_COUNT)
    results_objs = [pool.apply_async(async_episode, args=(best_model_num,)) for _ in range(SELF_GAMES)]
    pool.close()
    pool.join()

    for result in results_objs:
        result = result.get()
        states += result[0]
        valids += result[1]
        improv_policy += result[2]
        win_loss += result[3]

    return states, valids, improv_policy, win_loss


def async_episode(best_model_num) -> tuple:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')


    valids, states, improv_policy, win_loss = [], [], [], []
    best_model = CNNModel(best_model_num)
    best_model.load_model()
    board = chess.Board()
    mcts = MonteCarloTS(board.copy(), best_model)

    while not board.is_game_over() and board.fullmove_number < 150:
        move = mcts.search()
        board.push(move)
    reward_white = {"1-0": 1,
                    "1/2-1/2": 0,
                    "*": 0,
                    "0-1": -1}
    print(f'finished game with {board.result()}')
    for node in mcts.visited:
        policy = mcts.get_improved_policy(node, include_empty_spots=True)
        z = reward_white[board.result()]
        if node.state.board.turn == chess.BLACK:
            z *= -1
        states.append(node.state.get_representation())
        valids.append(node.state.get_valid_vector())
        improv_policy.append(policy)
        win_loss.append(z)
    return states, valids, improv_policy, win_loss


def bot_fight(best_model_num, new_model_num) -> int:
    new_model_wins = 0
    result_objs = []
    pool = mp.Pool(CPU_COUNT)
    for i in range(BOT_GAMES):
        result_objs += [pool.apply_async(async_arena, args=(i, best_model_num, new_model_num))]
    pool.close()
    pool.join()

    for result in result_objs:
        new_model_wins += result.get()
    return new_model_wins


def async_arena(iteration, best_model_num, new_model_num):
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) != 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    new_model_wins = 0
    board = chess.Board()
    best_model = CNNModel(best_model_num)
    best_model.load_model()
    new_model = CNNModel(new_model_num)
    new_model.load_model()
    mcts_best = MonteCarloTS(chess.Board(), best_model)
    mcts_new = MonteCarloTS(chess.Board(), new_model)
    if iteration % 2 == 0:
        turns = {"best": chess.WHITE,
                 "new": chess.BLACK}
    else:
        turns = {"best": chess.BLACK,
                 "new": chess.WHITE}
    while not board.is_game_over() and board.fullmove_number < 150 and not board.is_repetition(count=4):
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
    print(f'best model: {best_model_num}, new model won {new_model_wins}')
    return new_model_wins


if __name__ == "__main__":
    training_pipeline()
