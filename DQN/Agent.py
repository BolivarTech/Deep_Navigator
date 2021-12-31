import numpy as np
import random
import logging as log

from collections import deque

from UnityEnv import UnityEnv
from .QNNetwork_m3 import QNNetwork
from .ReplayBuffer import ReplayBuffer

import torch
import torch.nn.functional as functnl
import torch.optim as optim

logLevel_ = log.INFO


class Agent:
    """
    Agent that interacts with and learns from the environment.
    """

    def __init__(self, qnnmodel: QNNetwork, epsl_start=1.0, epsl_end=0.01, epsl_decay=0.995,
                 buffer_size=int(1e6), batch_size=512, gamma=0.999, tau=1e-3, learn_rate=5e-4, update_target=16,
                 device=None, log_handler=None):
        """
        Agent object Constructor.

        :param qnnmodel (QNNetwork) QNN model implementation to be used
        :param eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        :param eps_end (float): minimum value of epsilon
        :param eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        :param buffer_size (int): replay buffer size
        :param batch_size (int): training mini-batch size
        :param gamma (float): Q Training discount factor
        :param tau (float): soft update of target parameters
        :param learn_rate (fl=oat): learning rate
        :param update_target (int): how often to update the target network
        :param device: device where to load the network, if not specified the best device available will be selected
                       "cuda" or "cpu"
        :param log_Handler (handlers): Log handler to be used in the logging (Default is None)
        """

        global logLevel_

        # Set the error logger
        self.__logger = log.getLogger('Agent')
        # Add handler to logger
        self.__logHandler = log_handler
        if self.__logHandler is not None:
            self.__logger.addHandler(self.__logHandler)
        else:
            self.__logger.debug(f"logHandler NOT defined (001)")
        # Set Logger Lever
        self.__logger.setLevel(logLevel_)
        self.__epsl_start = epsl_start
        self.__epsl_end = epsl_end
        self.__epsl_decay = epsl_decay
        self.__epsl = self.__epsl_start  # epsilon, for epsilon-greedy action selection
        self.__buffer_size = buffer_size
        self.__batch_size = batch_size
        self.__gamma = gamma
        self.__tau = tau
        self.__learn_rate = learn_rate
        self.__update_target = update_target
        if device is None:
            self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.__device = device

        # Q-Network
        self.__qnnetwork_policy = qnnmodel.to(self.__device)
        self.__state_size = self.__qnnetwork_policy.get_state_size()
        self.__action_size = self.__qnnetwork_policy.get_action_size()
        self.__qnnetwork_target = type(self.__qnnetwork_policy)(self.__state_size, self.__action_size,
                                                                log_handler=self.__logHandler).to(self.__device)
        self.__qnnetwork_target.load_state_dict(self.__qnnetwork_policy.state_dict())
        self.__qnnetwork_target.eval()  # Not calculate gradients on target network
        self.__optimizer = optim.Adam(self.__qnnetwork_policy.parameters(), lr=self.__learn_rate)

        # Replay memory
        self.__buffer = ReplayBuffer(self.__action_size, self.__buffer_size, self.__batch_size,
                                     log_handler=self.__logHandler)
        # Initialize time step (for updating every update_target steps)
        self.__time_step = 0

    def __del__(self):
        """
        Class Destructor

        """
        del self.__state_size
        del self.__action_size
        del self.__device
        del self.__epsl_start
        del self.__epsl_end
        del self.__epsl_decay
        del self.__epsl
        del self.__buffer_size
        del self.__batch_size
        del self.__gamma
        del self.__tau
        del self.__learn_rate
        del self.__update_target
        del self.__qnnetwork_policy
        del self.__qnnetwork_target
        del self.__optimizer
        del self.__buffer
        del self.__time_step
        del self.__logHandler
        del self.__logger

    def __learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[numpy array]) tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(self.__device)
        actions = torch.from_numpy(actions).long().to(self.__device)
        rewards = torch.from_numpy(rewards).float().to(self.__device)
        next_states = torch.from_numpy(next_states).float().to(self.__device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.__device)

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.__qnnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (self.__gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.__qnnetwork_policy(states).gather(1, actions)

        # Compute loss
        loss = functnl.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()

        # ------------------- update target network ------------------- #
        # Learn every self.__update_target time steps.
        self.__time_step = (self.__time_step + 1) % self.__update_target
        if self.__time_step == 0:
            self.__soft_update(self.__qnnetwork_policy, self.__qnnetwork_target)
            self.__logger.debug(f"Learn Step Performed (002)")

    def __soft_update(self, local_model, target_model):
        """
        Soft update model parameters.

           θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model:  (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.__tau * local_param.data + (1.0 - self.__tau) * target_param.data)

    def learn_step(self, state, action, reward, next_state, done):
        """
        Perform one step on the learning process

        :param state: Environment current state
        :param action: Current Action
        :param reward: Reward received by the selected accion
        :param next_state: Environment next state
        :param done: Episode ended
        """
        # Save experience in replay memory
        self.__buffer.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, get random subset and learn
        if len(self.__buffer) >= self.__batch_size:
            experiences = self.__buffer.sample()
            self.__learn(experiences)

    def actions(self, state):
        """
        Returns action for given state as per current policy.

        :param state: (array_like): current state
        :return: Action selected by network's policy
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.__device)
        self.__qnnetwork_policy.eval()  # set policy network on eval mode
        with torch.no_grad():
            action_values = self.__qnnetwork_policy(state)
        self.__qnnetwork_policy.train()  # set policy network on training mode

        # Epsilon-greedy action selection
        if random.random() > self.__epsl:
            temp_act = action_values.cpu().data.numpy()
            self.__logger.debug(f"Action Values: {temp_act}")
            return np.argmax(temp_act).item()
        else:
            return random.choice(np.arange(self.__action_size))

    def training(self, envrm: UnityEnv, mean_score_update_timeout=20000, n_episodes=int(2e6), max_time_steps=int(1e10)):
        """
        Deep Q-Learning Agent Training

        :param envrm: Object that contain the training environment
        :param mean_score_update_timeout: maximum number of episodes where not improvement on the mean score was found
        :param n_episodes: (int) maximum number of training episodes
        :param max_time_steps: (int) maximum number of time steps per episode
        :return scores: [list] episodes' end scores
        """

        log.info(f"Training model on {self.__device} device")
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        mean_score = 0
        mean_score_update_timer = 0
        self.__epsl = self.__epsl_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):
            state = envrm.reset()
            score = envrm.get_score()
            for t in range(max_time_steps):
                action = self.actions(state)
                self.__logger.debug(f"Episode: {i_episode} Step: {t} Action: {action} (003)")
                next_state, reward, score, done = envrm.set_action(action)
                self.learn_step(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save episodes scores
            self.__epsl = max(self.__epsl_end, self.__epsl_decay * self.__epsl)  # decrease epsilon
            new_mean_score = np.mean(scores_window)
            self.__logger.info(f'Episode {i_episode:d}\tScore: {score:.2f}')
            if i_episode % 100 == 0:
                self.__logger.info(f'Episode {i_episode:d}\tAverage Score: {new_mean_score:.2f}')
            if new_mean_score > mean_score:
                torch.save(self.__qnnetwork_policy.state_dict(), 'checkpoint.pth')
                self.__logger.info(f'Environment checkpoint saved at {i_episode:d} episodes\tAverage Score: '
                                   f'{new_mean_score:.2f}')
                mean_score = new_mean_score
                mean_score_update_timer = 0
            if mean_score_update_timer > mean_score_update_timeout:
                break
            else:
                mean_score_update_timer += 1
        self.__logger.info(f'Best Environment checkpoint saved at Average Score: {mean_score:.2f}')
        return scores

    def save(self, file_name):
        """
        Save the network to file_name

        :param file_name:  File where to save network
        """
        torch.save(self.__qnnetwork_policy.state_dict(), file_name)

    def load(self, file_name, training_mode=True):
        """
        Load Network from file_name

        :param file_name: File from where to load network
        :param training_mode: if true the training target network is created, if false not to save memory
        :return: NamedTuple with missing_keys and unexpected_keys fields
        """
        load_result = self.__qnnetwork_policy.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
        self.__qnnetwork_policy.to(self.__device)
        self.__logger.debug(f"Load file {file_name} results:\n{load_result} (004)")
        if training_mode:
            self.__qnnetwork_target = type(self.__qnnetwork_policy)(self.__state_size, self.__action_size,
                                                                    log_handler=self.__logHandler).to(self.__device)
            self.__qnnetwork_target.load_state_dict(self.__qnnetwork_policy.state_dict())
            self.__qnnetwork_target.eval()  # Not calculate gradients on target network
        elif "__qnnetwork_target" in locals():
            del self.__qnnetwork_target
        return load_result
