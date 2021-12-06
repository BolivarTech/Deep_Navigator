import torch
import torch.nn as nn
import torch.nn.functional as functl

import numpy as np

import logging as log

logLevel_ = log.INFO


class QNNetwork(nn.Module):
    """
    Q-Network Model.
    """

    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=64, fc3_units=64, device=None, log_handler=None):
        """
        Q-Network Constructor.

        :param state_size (int): Dimension of each state
        :param action_size (int): Dimension of each action
        :param fc1_units (int): Number of nodes in first hidden layer
        :param fc2_units (int): Number of nodes in second hidden layer
        :param fc3_units (int): Number of nodes in second hidden layer
        :param device: device where to load the network, if not specified the best device available will be selected
                       "cuda" or "cpu"
        :param log_Handler (handlers): Log handler to be used in the logging (Default is None)
        """

        global logLevel_

        super(QNNetwork, self).__init__()
        # Set the error logger
        self.__logger = log.getLogger('QNNetwork')
        # Add handler to logger
        if log_handler is not None:
            self.__logger.addHandler(log_handler)
        else:
            self.__logger.debug(f"logHandler NOT defined (001)")
        # Set Logger Lever
        self.__logger.setLevel(logLevel_)
        self.__state_size = state_size
        self.__action_size = action_size
        self.fc1 = nn.Linear(self.__state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, self.__action_size)
        if device is None:
            self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.__device = device
        self.to(self.__device)

    def __del__(self):
        """
        Class Destructor

        """
        del self.fc1
        del self.fc2
        del self.fc3
        del self.fc4
        del self.__device
        del self.__state_size
        del self.__action_size
        del self.__logger

    def forward(self, state):
        """
        Forward propagate the network to maps state to action values.

        :param state: Environment current state
        """
        x = functl.relu(self.fc1(state))
        x = functl.relu(self.fc2(x))
        x = functl.relu(self.fc3(x))
        x = functl.softmax(self.fc4(x),dim=1)
        return x

    def set_state_size(self, state_size):
        """
        Set the model states size

        :param: state_size
        """
        self.__state_size = state_size

    def get_state_size(self):
        """
        Returns the model states size

        :return: model states size
        """
        return self.__state_size

    def set_action_size(self, action_size):
        """
        Set the model action size

        :param: action_size
        """
        self.__action_size = action_size

    def get_action_size(self):
        """
        Returns the model actions size

        :return: model actions size
        """
        return self.__action_size

    def save(self, file_name):
        """
        Save the network to file_name

        :param file_name:  File where to save network
        """
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        """
        Load Network from file_name

        :param file_name: File from where to load network
        :return: NamedTuple with missing_keys and unexpected_keys fields
        """

        load_result = self.load_state_dict(torch.load(file_name,map_location=torch.device('cpu')))
        self.to(self.__device)
        self.__logger.debug(f"Load file {file_name} results:\n{load_result} (002)")
        return load_result

    def set_grad_enabled(self, mode):
        """
        Enable ot desable the torch gradien calculation
        :param mode: True or False to enable or disable gradien calculation
        """

        torch.set_grad_enabled(mode)

    def get_action(self, state):
        """
        Get the best greedy action based on the given state

        :param state: Current state
        :return: Best greedy action index
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.__device)
        action_values = self(state)
        temp_act = action_values.cpu().data.numpy()
        self.__logger.debug(f"Action Values: {temp_act}")
        return np.argmax(temp_act).item()
