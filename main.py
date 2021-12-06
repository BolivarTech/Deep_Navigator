"""
Banana Collector

By: Julian Bolivar
Version: 1.0.0
"""
from collections import deque

from UnityEnv import UnityEnv
from DQN import Agent, QNNetwork

from argparse import ArgumentParser
import logging as log
import logging.handlers
import sys
import os

import numpy as np

import csv
from datetime import datetime


# Main Logger
logHandler = None
logLevel_ = logging.INFO
# OS running
OS_ = 'unknown'

# Number of plays
num_episodes = 13


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", required=False, action="store_true", help="Perform a model training, if -m "
                                                                                   "not specified a new model is "
                                                                                   "trained.")
    parser.add_argument("-p", "--play", required=False, action="store_true", help="Perform the model playing.")
    parser.add_argument("-m", "--model", required=False, type=str, default=None,
                        help="Path to an pytorch model file.")
    return parser


def save_scores(scores):
    """

    :param scores:
    :return:
    """

    with open(f'scores-{datetime.now().strftime("%Y%m%d%H:%M:%S")}.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(scores)


def train(model_file):
    """
    Setup the training environment

    :param model_file: file path with the model to be loaded
    :return: None
    """

    global logHandler
    global OS_
    global logLevel_

    u_env = None
    if OS_ == 'linux':
        u_env = UnityEnv(train_mode=True, env_filename="./Banana_Linux/Banana.x86_64", log_handler=logHandler)
    elif OS_ == 'win32':
        u_env = UnityEnv(train_mode=True, env_filename=".\\Banana_Windows_x86_64\\Banana.exe", log_handler=logHandler)
    log.info(f"Unity Environmet {OS_} loaded (001)")
    # number of agents in the environment
    log.info(f'Number of agents: {u_env.get_num_agents()}')
    # number of actions
    log.info(f'Number of actions: {u_env.get_num_actions()}')
    # examine the state space
    log.info(f'States look like: {u_env.get_state()}')
    log.info(f'States have length: {u_env.get_state_size()}')
    # Generate the QNNetwork
    qnn = QNNetwork(u_env.get_state_size(), u_env.get_num_actions(), log_handler=logHandler)
    if model_file is not None:
        qnn.load(model_file)
    # train the agent
    agn = Agent(qnn, log_handler=logHandler)
    scores = agn.training(u_env)
    save_scores(scores)


def play(model_file):
    """
    Perform an play using the agent
    :param model_file: file path with the model to be loaded
    """

    global logHandler
    global OS_
    global logLevel_
    global num_episodes

    scores_window = deque(maxlen=num_episodes)  # last num_plays scores
    u_env = None
    if OS_ == 'linux':
        u_env = UnityEnv(train_mode=False, env_filename="./Banana_Linux/Banana.x86_64", log_handler=logHandler)
    elif OS_ == 'win32':
        u_env = UnityEnv(train_mode=False, env_filename=".\\Banana_Windows_x86_64\\Banana.exe", log_handler=logHandler)
    log.info(f"Unity Environmet {OS_} loaded (001)")
    # Generate the QNNetwork
    qnn = QNNetwork(u_env.get_state_size(), u_env.get_num_actions(), log_handler=logHandler)
    if model_file is not None:
        qnn.load(model_file)
    else:
        log.error(f"Can't Play because a model file wasn't specified")
        return
    qnn.eval()  # set policy network on eval mode
    qnn.set_grad_enabled(False)
    for i_episode in range(1, num_episodes + 1):
        state = u_env.reset()
        score = u_env.get_score()
        done = False
        t = 0
        while not done:
            action = qnn.get_action(state)
            log.debug(f"Episode: {i_episode} Step: {t} Action: {action} (003)")
            state, _, score, done = u_env.set_action(action)
            t += 1
        scores_window.append(score)  # save most recent score
    mean_score = np.mean(scores_window)
    print(f"The mean score was {mean_score:.2f} over {num_episodes} episodes ")


def main():
    """
     Run the main function

    """

    args = build_argparser().parse_args()
    model_file = args.model
    if args.train and args.play:
        log.error("Options Train and Play can't be used togethers")
    elif args.train:
        if model_file is not None:
            log.debug(f"Training option selected with file {model_file}")
        else:
            log.debug(f"Training option selected with new model")
        train(model_file)
    elif args.play:
        if model_file is not None:
            log.debug(f"Play option selected with file {model_file}")
            play(model_file)
        else:
            log.error(f"On Play option a model file must be specified")
    else:
        log.debug(f"Not option was selected with file {model_file}")


if __name__ == '__main__':

    loggPath = "."
    LogFileName = loggPath + '/' + 'DNav.log'
    # Check where si running
    if sys.platform.startswith('freebsd'):
        OS_ = 'freebsd'
    elif sys.platform.startswith('linux'):
        OS_ = 'linux'
    elif sys.platform.startswith('win32'):
        OS_ = 'win32'
    elif sys.platform.startswith('cygwin'):
        OS_ = 'cygwin'
    elif sys.platform.startswith('darwin'):
        OS_ = 'darwin'
    if OS_ == 'linux':
        loggPath = '/var/log/DNav'
        LogFileName = loggPath + '/' + 'DNav.log'
    elif OS_ == 'win32':
        loggPath = os.getenv('LOCALAPPDATA') + '\\DNav'
        LogFileName = loggPath + '\\' + 'DNav.log'

    # Configure the logger
    os.makedirs(loggPath, exist_ok=True)  # Create log path
    logger = log.getLogger('DNav')  # Get Logger
    # Add the log message file handler to the logger
    logHandler = log.handlers.RotatingFileHandler(LogFileName, maxBytes=10485760, backupCount=5)
    # Logger Formater
    logFormatter = log.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                                     datefmt='%Y/%m/%d %H:%M:%S')
    logHandler.setFormatter(logFormatter)
    # Add handler to logger
    if 'logHandler' in globals():
        logger.addHandler(logHandler)
    else:
        logger.debug(f"logHandler NOT defined (009)")
        # Set Logger Lever
    # logger.setLevel(logging.INFO)
    logger.setLevel(logLevel_)
    # Start Running
    logger.debug(f"Running in {OS_} (001)")
    main()
    exit(0)
