# ChessAI

Zero Knowledge Chess AI based off alpha-zero. Uses a neural network and Monte Carlo Tree Search to reduce search depth.

Three main training phases for neural network:
1. Gather data through self-play
2. Train new network
3. Evaluate whether new network is better, replace with the new one if so.


To train:

1. In terminal/cmd, navigate to folder in which you want to clone repo 
2. Clone Repo
3. Run `pip3 install -r requirements.txt`
4. Run `training_pipline.py`

By default, one core on your PC is left free.
